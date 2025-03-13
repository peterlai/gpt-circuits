import { atom } from "jotai";
import { selectAtom } from "jotai/utils";

import { BlockData, BlockFeatureData, blocksAtom } from "./Block";
import { SelectionState, selectionStateAtom } from "./Selection";

class ConnectionData {
  public upstreamLayerIdx: number;
  public upstreamTokenOffset: number;
  public downstreamTokenOffset: number;
  public ablations: AblationData[] = [];
  // Importance of the upstream block to the downstream block - [0, 1] where 1 is most important.
  public importance: number = 0;

  constructor(
    upstreamLayerIdx: number,
    upstreamTokenOffset: number,
    downstreamTokenOffset: number,
    importance: number
  ) {
    this.upstreamLayerIdx = upstreamLayerIdx;
    this.upstreamTokenOffset = upstreamTokenOffset;
    this.downstreamTokenOffset = downstreamTokenOffset;
    this.importance = importance;
  }

  get downstreamLayerIdx() {
    return ConnectionData.getDownstreamLayerIdx(this.upstreamLayerIdx);
  }

  get key() {
    return ConnectionData.getKey(
      this.upstreamLayerIdx,
      this.upstreamTokenOffset,
      this.downstreamTokenOffset
    );
  }

  static getDownstreamLayerIdx(upstreamLayerIdx: number) {
    return upstreamLayerIdx + 1;
  }

  static getKey(
    upstreamLayerIdx: number,
    upstreamTokenOffset: number,
    downstreamTokenOffset: number
  ) {
    const downstreamLayerIdx = ConnectionData.getDownstreamLayerIdx(upstreamLayerIdx);
    const upstreamKey = `${upstreamTokenOffset}.${upstreamLayerIdx}`;
    const downstreamKey = `${downstreamTokenOffset}.${downstreamLayerIdx}`;
    return `${upstreamKey} -> ${downstreamKey}`;
  }
}

class AblationData {
  public upstreamLayerIdx: number;
  public upstreamTokenOffset: number;
  public upstreamFeatureId: number;
  public downstreamTokenOffset: number;
  public downstreamFeatureId: number;
  public weight: number;

  constructor(
    upstreamLayerIdx: number,
    upstreamTokenOffset: number,
    upstreamFeatureId: number,
    downstreamTokenOffset: number,
    downstreamFeatureId: number,
    weight: number
  ) {
    this.upstreamLayerIdx = upstreamLayerIdx;
    this.upstreamTokenOffset = upstreamTokenOffset;
    this.upstreamFeatureId = upstreamFeatureId;
    this.downstreamTokenOffset = downstreamTokenOffset;
    this.downstreamFeatureId = downstreamFeatureId;
    this.weight = weight;
  }

  get downstreamLayerIdx() {
    return this.upstreamLayerIdx + 1;
  }

  get upstreamBlockKey() {
    return BlockData.getKey(this.upstreamTokenOffset, this.upstreamLayerIdx);
  }

  get downstreamBlockKey() {
    return BlockData.getKey(this.downstreamTokenOffset, this.downstreamLayerIdx);
  }

  get upstreamFeatureKey() {
    return BlockFeatureData.getKey(
      this.upstreamTokenOffset,
      this.upstreamLayerIdx,
      this.upstreamFeatureId
    );
  }

  get downstreamFeatureKey() {
    return BlockFeatureData.getKey(
      this.downstreamTokenOffset,
      this.downstreamLayerIdx,
      this.downstreamFeatureId
    );
  }
}

// Represents modifiers to apply to a specific feature
class ConnectionModifier {
  public weight: number = 0; // Line opacity
  public width: number = 0; // Line width
  public isGray: boolean = false;

  constructor(connection: ConnectionData, selectionState: SelectionState) {
    this.weight = 1;
    this.width = 1;

    // Is this connection attached to anything with focus?
    const isAnythingFocused = !!selectionState.focusedBlock || !!selectionState.focusedFeature;
    let hasBlockFocus = false;
    let hasFeatureFocus = false;
    let focusedAblations: AblationData[] = [];
    if (selectionState.focusedBlock) {
      if (
        (selectionState.focusedBlock.layerIdx === connection.upstreamLayerIdx &&
          selectionState.focusedBlock.tokenOffset === connection.upstreamTokenOffset) ||
        (selectionState.focusedBlock.layerIdx === connection.downstreamLayerIdx &&
          selectionState.focusedBlock.tokenOffset === connection.downstreamTokenOffset)
      ) {
        hasBlockFocus = true;
      }
    } else if (selectionState.focusedFeature) {
      focusedAblations = connection.ablations.filter(
        (ablation) =>
          ablation.upstreamFeatureKey === selectionState.focusedFeature?.key ||
          ablation.downstreamFeatureKey === selectionState.focusedFeature?.key
      );
      hasFeatureFocus = focusedAblations.length > 0;
    }

    if (hasFeatureFocus) {
      // Emphasize connections related to the focused feature.
      const ablationWeights = focusedAblations.map((ablation) => ablation.weight);
      const maxAblationWeight = Math.max(...ablationWeights);
      if (maxAblationWeight > 0.9) {
        this.width = 2;
        this.weight = 3;
      } else if (maxAblationWeight > 0.5) {
        this.width = 1;
        this.weight = 3;
      } else if (maxAblationWeight > 0.1) {
        this.width = 1;
        this.weight = 2;
      } else {
        this.width = 1;
        this.weight = 1;
      }
    } else if (hasBlockFocus || !isAnythingFocused) {
      // Show default connection weight and width.
      if (connection.importance > 0.9) {
        this.width = 2;
        this.weight = 3;
      } else if (connection.importance > 0.5) {
        this.width = 1;
        this.weight = 2;
      } else if (connection.importance > 0.1) {
        this.width = 1;
        this.weight = 1;
      } else {
        this.width = 1;
        this.weight = 1;
      }
    } else {
      // Gray out connection if not related to any focused block/feature.
      this.isGray = true;
    }
  }

  // Used to avoid re-rendering when the feature modifier hasn't changed
  static areEqual(a: ConnectionModifier, b: ConnectionModifier) {
    return a.weight === b.weight && a.width === b.width && a.isGray === b.isGray;
  }
}

// Connection data parsed from block data
const connectionsAtom = atom((get) => {
  const blocks = get(blocksAtom);
  const connections: { [key: string]: ConnectionData } = {};

  // Create connections and add ablation weights between pairs of features
  for (const block of Object.values(blocks)) {
    for (const feature of Object.values(block.features)) {
      for (const upstreamAblation of Object.values(feature.ablatedBy)) {
        // Get or create connection
        const connectionKey = ConnectionData.getKey(
          upstreamAblation.layerIdx,
          upstreamAblation.tokenOffset,
          feature.tokenOffset
        );
        const connection =
          connections[connectionKey] ||
          new ConnectionData(
            upstreamAblation.layerIdx,
            upstreamAblation.tokenOffset,
            feature.tokenOffset,
            block.upstreamImportances[upstreamAblation.tokenOffset]
          );
        connections[connectionKey] = connection;
        // Add ablation to connection
        const ablation = new AblationData(
          upstreamAblation.layerIdx,
          upstreamAblation.tokenOffset,
          upstreamAblation.featureId,
          feature.tokenOffset,
          feature.featureId,
          upstreamAblation.weight
        );
        connection.ablations.push(ablation);
      }
    }
  }

  return connections;
});

// Creates a connection modifier atom for a specific connection
function createConnectionModifierAtom(connection: ConnectionData) {
  return selectAtom(
    selectionStateAtom,
    (selectionState) => {
      return new ConnectionModifier(connection, selectionState);
    },
    ConnectionModifier.areEqual
  );
}

export { ConnectionData, connectionsAtom, createConnectionModifierAtom };
