import { atom } from "jotai";
import { atomWithHash } from "jotai-location";

import { getInspectSamplePath } from "../views/App/urls";
import { BlockData, BlockFeatureData } from "./Block";
import { alignmentAtom, AlignmentOptions, searchQueryAtom } from "./Search";

interface SelectionState {
  hoveredBlock: BlockData | null;
  selectedBlock: BlockData | null;
  focusedBlock: BlockData | null;
  hoveredFeature: BlockFeatureData | null;
  selectedFeature: BlockFeatureData | null;
  focusedFeature: BlockFeatureData | null;
  hoveredUpstreamOffset: number | null;
}

const hashSelectionAtom = atomWithHash("selection", "", {
  setHash: (searchParams: string) => {
    // NOTE: Changing the URL hash triggers re-render. I'm not sure if this can be avoided.
    const selectionKey = new URLSearchParams(searchParams).get("selection")?.replaceAll('"', "");
    const [modelId, sampleId, version] = window.location.hash
      ?.replace("#", "")
      .split("/")
      .slice(1, 4);
    const url = `#${getInspectSamplePath(modelId, sampleId, version, selectionKey)}`;
    window.history.replaceState(null, "", url);
  },
});

const hoveredBlockAtom = atom<BlockData | null>(null);
const selectedBlockAtom = atom<BlockData | null>(null);
const hoveredFeatureAtom = atom<BlockFeatureData | null>(null);
const selectedFeatureAtom = atom<BlockFeatureData | null>(null);
const hoveredUpstreamOffsetAtom = atom<number | null>(null);

// Atom for toggling the selected block or feature.
const toggleSelectionAtom = atom(null, (get, set, object: BlockData | BlockFeatureData | null) => {
  let selectedBlock = get(selectedBlockAtom);
  let selectedFeature = get(selectedFeatureAtom);
  let selectedObject = selectedBlock || selectedFeature;

  // If the selection should change, update selections.
  if (selectedObject !== object) {
    let hashKey: string;
    if (object instanceof BlockData) {
      set(selectedBlockAtom, object);
      set(selectedFeatureAtom, null);
      hashKey = object.key;
    } else if (object instanceof BlockFeatureData) {
      set(selectedBlockAtom, null);
      set(selectedFeatureAtom, object);
      hashKey = object.key;
    } else {
      set(selectedBlockAtom, null);
      set(selectedFeatureAtom, null);
      hashKey = "";
    }

    // Clear search options
    set(searchQueryAtom, "");
    set(alignmentAtom, AlignmentOptions.Token);

    // Update the URL location
    set(hashSelectionAtom, (f) => hashKey);
  }
});

// Represents the current selection state
const selectionStateAtom = atom<SelectionState>((get) => {
  const hoveredBlock = get(hoveredBlockAtom);
  const selectedBlock = get(selectedBlockAtom);
  const hoveredFeature = get(hoveredFeatureAtom);
  const selectedFeature = get(selectedFeatureAtom);
  const hoveredUpstreamOffset = get(hoveredUpstreamOffsetAtom);

  // Focus is on the hovered feature if it exists, otherwise the selected feature
  const focusedFeature = hoveredFeature || selectedFeature;
  // Block can only be focused if no feature is focused.
  const focusedBlock = focusedFeature ? null : selectedBlock;

  return {
    hoveredBlock: hoveredBlock,
    selectedBlock: selectedBlock,
    focusedBlock: focusedBlock,
    hoveredFeature: hoveredFeature,
    selectedFeature: selectedFeature,
    focusedFeature: focusedFeature,
    hoveredUpstreamOffset: hoveredUpstreamOffset,
  };
});

export {
  hoveredBlockAtom,
  hoveredFeatureAtom,
  hoveredUpstreamOffsetAtom,
  selectedBlockAtom,
  selectedFeatureAtom,
  selectionStateAtom,
  toggleSelectionAtom,
};
export type { SelectionState };
