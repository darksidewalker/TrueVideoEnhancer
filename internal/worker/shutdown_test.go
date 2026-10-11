package worker

import (
	"context"
	"testing"
	"time"
)

func TestShutdownCancelsActiveAndQueuedJobsAndWaitsForWorker(t *testing.T) {
	m := NewManager(Config{})
	started := make(chan struct{})
	cancelled := make(chan struct{})
	release := make(chan struct{})
	exited := make(chan struct{})
	m.runJob = func(ctx context.Context, id string, args []string) {
		close(started)
		<-ctx.Done()
		close(cancelled)
		<-release
		close(exited)
	}
	dir := t.TempDir()
	first, err := m.Start(makeTestRequest(t, dir, "first.mp4"))
	if err != nil {
		t.Fatal(err)
	}
	<-started
	second, err := m.Start(makeTestRequest(t, dir, "second.mp4"))
	if err != nil {
		t.Fatal(err)
	}
	ctx, cancel := context.WithTimeout(context.Background(), time.Second)
	defer cancel()
	done := make(chan error, 1)
	go func() { done <- m.Shutdown(ctx) }()
	select {
	case <-cancelled:
	case <-ctx.Done():
		t.Fatal("active job was not cancelled")
	}
	if _, err = m.Start(makeTestRequest(t, dir, "third.mp4")); err == nil {
		t.Error("accepted new work during shutdown")
	}
	select {
	case err := <-done:
		t.Fatalf("shutdown returned before worker cleanup: %v", err)
	default:
	}
	close(release)
	if err = <-done; err != nil {
		t.Fatal(err)
	}
	<-exited
	for _, id := range []string{first.ID, second.ID} {
		job, ok := m.Get(id)
		if !ok || job.Status != "cancelled" {
			t.Fatalf("job %s not cancelled: %+v", id, job)
		}
	}
	if err = m.Shutdown(ctx); err != nil {
		t.Fatalf("repeated shutdown failed: %v", err)
	}
}

func TestShutdownRespectsDeadline(t *testing.T) {
	m := NewManager(Config{})
	started := make(chan struct{})
	release := make(chan struct{})
	m.runJob = func(context.Context, string, []string) { close(started); <-release }
	_, err := m.Start(makeTestRequest(t, t.TempDir(), "input.mp4"))
	if err != nil {
		t.Fatal(err)
	}
	<-started
	ctx, cancel := context.WithCancel(context.Background())
	cancel()
	if err = m.Shutdown(ctx); err != context.Canceled {
		t.Fatalf("expected context deadline/cancellation, got %v", err)
	}
	close(release)
	if err = m.Shutdown(context.Background()); err != nil {
		t.Fatal(err)
	}
}
