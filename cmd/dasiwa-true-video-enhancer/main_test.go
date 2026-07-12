package main

import "testing"

func TestShouldOpenBrowserByDefault(t *testing.T) {
	t.Setenv("DASIWA_NO_BROWSER", "")
	if !shouldOpenBrowser() {
		t.Fatal("browser should open by default")
	}
}

func TestShouldOpenBrowserCanBeDisabled(t *testing.T) {
	t.Setenv("DASIWA_NO_BROWSER", "1")
	if shouldOpenBrowser() {
		t.Fatal("browser should stay closed when DASIWA_NO_BROWSER=1")
	}
}
