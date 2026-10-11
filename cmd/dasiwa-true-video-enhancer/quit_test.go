package main

import (
	"bytes"
	"context"
	"encoding/json"
	"net"
	"net/http"
	"os"
	"os/exec"
	"strconv"
	"testing"
	"time"
)

// Exercise run() in an owned subprocess so a successful HTTP response is not
// mistaken for process termination. No existing app or fixed port is touched.
func TestQuitActuallyExitsAppProcess(t *testing.T) {
	listener, err := net.Listen("tcp", "127.0.0.1:0")
	if err != nil {
		t.Fatal(err)
	}
	port := listener.Addr().(*net.TCPAddr).Port
	listener.Close()
	ctx, cancel := context.WithTimeout(context.Background(), 15*time.Second)
	defer cancel()
	cmd := exec.CommandContext(ctx, os.Args[0], "-test.run=^TestQuitProcessHelper$")
	cmd.Env = append(os.Environ(), "TVE_QUIT_TEST_HELPER=1", "DASIWA_NO_BROWSER=1", "DASIWA_PORT="+strconv.Itoa(port))
	var output bytes.Buffer
	cmd.Stdout = &output
	cmd.Stderr = &output
	if err = cmd.Start(); err != nil {
		t.Fatal(err)
	}
	done := make(chan struct{})
	var waitErr error
	go func() { waitErr = cmd.Wait(); close(done) }()
	t.Cleanup(func() { cancel(); <-done })
	base := "http://127.0.0.1:" + strconv.Itoa(port)
	client := http.Client{Timeout: time.Second}
	ready := false
	for deadline := time.Now().Add(5 * time.Second); time.Now().Before(deadline); {
		resp, e := client.Get(base + "/api/health")
		if e == nil {
			resp.Body.Close()
			if resp.StatusCode == http.StatusOK {
				ready = true
				break
			}
		}
		select {
		case <-done:
			t.Fatalf("app exited before ready: %v\n%s", waitErr, output.String())
		case <-time.After(20 * time.Millisecond):
		}
	}
	if !ready {
		t.Fatal("app never became ready")
	}
	resp, err := client.Post(base+"/api/quit", "application/json", nil)
	if err != nil {
		t.Fatal(err)
	}
	var body map[string]string
	err = json.NewDecoder(resp.Body).Decode(&body)
	resp.Body.Close()
	if err != nil || resp.StatusCode != http.StatusOK || body["status"] != "shutting-down" {
		t.Fatalf("quit response: %d %+v %v", resp.StatusCode, body, err)
	}
	select {
	case <-done:
		if waitErr != nil {
			t.Fatalf("app failed on quit: %v\n%s", waitErr, output.String())
		}
	case <-time.After(12 * time.Second):
		t.Fatal("Quit acknowledged but app process did not exit")
	}
	conn, err := net.DialTimeout("tcp", "127.0.0.1:"+strconv.Itoa(port), time.Second)
	if err == nil {
		conn.Close()
		t.Fatal("listener still open after app exit")
	}
}

func TestQuitProcessHelper(t *testing.T) {
	if os.Getenv("TVE_QUIT_TEST_HELPER") != "1" {
		return
	}
	if err := run(); err != nil {
		t.Fatal(err)
	}
}
