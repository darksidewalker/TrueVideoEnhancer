package runtimecheck

import "testing"

func TestBuiltInUpscalersStartWithNativeRealtimePairPerContentType(t *testing.T) {
	models := builtInModelChoices(t.TempDir())
	want := []struct {
		id, subcategory, scale string
	}{
		{"animejanai-hd-v3-compact-2x", "anime", "2x"},
		{"anime-sharp-v4-2x", "anime", "2x"},
		{"nomosuni-span-anime-4x", "anime", "4x"},
		{"animejanai-hd-v3-compact-mixed-2x", "mixed", "2x"},
		{"nomosuni-span-2x", "mixed", "2x"},
		{"ultrasharpv2-4x", "mixed", "4x"},
		{"realplksr-restoration-2x", "realism", "2x"},
		{"clearreality-4x", "realism", "4x"},
	}
	positions := make(map[string]int)
	for i, model := range models {
		positions[model.ID] = i
	}
	for _, item := range want {
		pos, ok := positions[item.id]
		if !ok {
			t.Fatalf("missing recommended model %q", item.id)
		}
		model := models[pos]
		if model.SubCategory != item.subcategory {
			t.Errorf("%s subcategory = %q, want %q", item.id, model.SubCategory, item.subcategory)
		}
		if len(model.File) < 2 || model.File[:2] != item.scale {
			t.Errorf("%s file %q does not have native %s scale", item.id, model.File, item.scale)
		}
	}
}
