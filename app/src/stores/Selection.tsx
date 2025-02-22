import { atom } from "jotai";
import { atomWithHash } from "jotai-location";

import { getInspectSamplePath } from "../views/App/urls";
import { BlockFeatureData } from "./Block";
import { alignmentAtom, AlignmentOptions, searchQueryAtom } from "./Search";

interface FeatureSelections {
  hoveredFeature: BlockFeatureData | null;
  selectedFeature: BlockFeatureData | null;
  focusedFeature: BlockFeatureData | null;
  hoveredUpstreamOffset: number | null;
}

const hashFeatureAtom = atomWithHash("feature", "", {
  setHash: (searchParams: string) => {
    // NOTE: Changing the URL hash triggers re-render. I'm not sure if this can be avoided.
    const featureKey = new URLSearchParams(searchParams).get("feature")?.replaceAll('"', "");
    const [modelId, sampleId] = window.location.hash?.replace("#", "").split("/").slice(1, 3);
    window.history.replaceState(
      null,
      "",
      `#${getInspectSamplePath(modelId, sampleId, featureKey)}`
    );
  },
});
const hoveredFeatureAtom = atom<BlockFeatureData | null>(null);
const selectedFeatureAtom = atom<BlockFeatureData | null>(null);
const hoveredUpstreamOffsetAtom = atom<number | null>(null);

// Atom for toggling the selected feature
const toggleSelectedFeatureAtom = atom(null, (get, set, feature: BlockFeatureData | null) => {
  const selectedFeature = get(selectedFeatureAtom);

  if (feature !== selectedFeature) {
    // Update the selected feature
    set(selectedFeatureAtom, feature);

    // Clear search options
    set(searchQueryAtom, "");
    set(alignmentAtom, AlignmentOptions.Token);

    // Update the URL location
    const featureKey = feature ? feature.key : "";
    set(hashFeatureAtom, (f) => featureKey);
  }
});

// Represents the current feature selection state
const featureSelectionsAtom = atom<FeatureSelections>((get) => {
  const hoveredFeature = get(hoveredFeatureAtom);
  const selectedFeature = get(selectedFeatureAtom);
  const hoveredUpstreamOffset = get(hoveredUpstreamOffsetAtom);

  // Focus is on the hovered feature if it exists, otherwise the selected feature
  const focusedFeature = hoveredFeature || selectedFeature;
  return {
    hoveredFeature: hoveredFeature,
    selectedFeature: selectedFeature,
    focusedFeature: focusedFeature,
    hoveredUpstreamOffset: hoveredUpstreamOffset,
  };
});

export {
  featureSelectionsAtom,
  hoveredFeatureAtom,
  hoveredUpstreamOffsetAtom,
  selectedFeatureAtom,
  toggleSelectedFeatureAtom,
};
export type { FeatureSelections };
