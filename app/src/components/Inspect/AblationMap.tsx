import classNames from "classnames";
import { useAtom, useAtomValue, useSetAtom } from "jotai";
import { useEffect, useMemo, useRef } from "react";

import {
  BlockData,
  BlockFeatureData,
  blocksAtom,
  createBlockModifierAtom,
} from "../../stores/Block";
import {
  ConnectionData,
  connectionsAtom,
  createConnectionModifierAtom,
} from "../../stores/Connection";
import { createFeatureModifierAtom, createFeatureProfileAtom } from "../../stores/Feature";
import {
  contextLengthAtom,
  modelIdAtom,
  numLayersAtom,
  sampleIdAtom,
  sampleTokensAtom,
  targetIdxAtom,
} from "../../stores/Graph";
import { hoveredBlockAtom, hoveredFeatureAtom, toggleSelectionAtom } from "../../stores/Selection";
import { getInspectSamplePath } from "../../views/App/urls";

import "@fontsource/open-sans";
import "@fontsource/open-sans/300.css";
import "@fontsource/open-sans/500.css";
import "@fontsource/open-sans/700.css";
import "./AblationMap.scss";

function AblationMap({
  selectionKeyFromUrl,
  isEmbedded,
}: {
  selectionKeyFromUrl?: string;
  isEmbedded?: boolean;
}) {
  const targetIdx = useAtomValue(targetIdxAtom);
  const contextLength = useAtomValue(contextLengthAtom);
  const numLayers = useAtomValue(numLayersAtom);
  const blocks = useAtomValue(blocksAtom);
  const toggleSelection = useSetAtom(toggleSelectionAtom);
  const mapRef = useRef(null);

  useEffect(() => {
    if (!mapRef.current) return;
    const mapHolderEl = (mapRef.current as HTMLDivElement).parentElement;
    let resizeObserver: ResizeObserver | null = null;

    if (mapHolderEl) {
      // Scroll to target block upon render.
      let clientWidth = mapHolderEl.clientWidth;
      const scrollWidth = mapHolderEl.scrollWidth;
      mapHolderEl.scrollLeft = (scrollWidth / contextLength) * (targetIdx + 1) - clientWidth;

      // Align the map to the right edge of its original window upon window resize.
      resizeObserver = new ResizeObserver(() => {
        // Update the offset width and calculate the delta.
        let widthDelta = mapHolderEl.clientWidth - clientWidth;
        clientWidth = mapHolderEl.clientWidth;
        // Preserve the scroll position.
        let scrollLeftValue = mapHolderEl.scrollLeft - widthDelta;
        scrollLeftValue = Math.min(scrollLeftValue, scrollWidth - clientWidth);
        scrollLeftValue = Math.max(scrollLeftValue, 0);
        mapHolderEl.scrollLeft = scrollLeftValue;
      });
      resizeObserver.observe(mapHolderEl);
    }

    // Clean up.
    return () => {
      if (resizeObserver) {
        resizeObserver.disconnect();
      }
    };
  }, [contextLength, targetIdx]);

  // Set the selected block or feature from the URL if it exists.
  useEffect(() => {
    if (selectionKeyFromUrl) {
      const [tokenOffset, layerIdx, featureIdx] = selectionKeyFromUrl.split(".").map(Number);
      const block = blocks[BlockData.getKey(tokenOffset, layerIdx)];
      const objectToSelect = featureIdx ? block?.features[selectionKeyFromUrl] : block;
      toggleSelection(objectToSelect);

      // Make sure the selected layer is in view.
      const mapHolderEl = mapRef.current ? (mapRef.current as HTMLDivElement).parentElement : null;
      if (!mapHolderEl) return;
      let clientHeight = mapHolderEl.clientHeight;
      const scrollHeight = mapHolderEl.scrollHeight;
      const targetScrollTop = (scrollHeight / numLayers) * (layerIdx + 1) - clientHeight;
      if (targetScrollTop > 0 && mapHolderEl.scrollTop < targetScrollTop) {
        mapHolderEl.scrollTop = targetScrollTop;
      }
    }
  }, [selectionKeyFromUrl, blocks, toggleSelection, numLayers]);

  // Set CSS variables
  const style = {
    "--context-length": contextLength,
    "--num-layers": numLayers,
  };

  return (
    <section
      id="Map"
      className={classNames({
        embedded: isEmbedded,
      })}
      style={style as React.CSSProperties}
      ref={mapRef}
      onClick={() => {
        // Clear selection
        toggleSelection(null);
      }}
    >
      <SampleText />
      <Blocks isEmbedded={isEmbedded || false} />
      <Connections />
    </section>
  );
}

function SampleText() {
  const sampleTokens = useAtomValue(sampleTokensAtom);
  const targetIdx = useAtomValue(targetIdxAtom);

  const isWords = Math.max(...sampleTokens.map((t) => t.length)) > 1;

  return (
    <>
      {sampleTokens.map((token, i) => {
        let text = token.replaceAll("\n", "⏎");
        return (
          <div
            key={i}
            className={classNames({
              cell: true,
              input: true,
              target: i === targetIdx,
            })}
            style={{ gridColumn: i + 1, gridRow: 1 }}
          >
            <div
              className={classNames({
                token: true,
                past: i < targetIdx,
                future: i > targetIdx,
                return: token === "\n",
                char: !isWords,
              })}
            >
              {text}
            </div>
          </div>
        );
      })}
    </>
  );
}

function Blocks({ isEmbedded }: { isEmbedded: boolean }) {
  const blocks = useAtomValue(blocksAtom);

  return (
    <>
      {Object.values(blocks)
        .sort(BlockData.compare)
        .map((block) => (
          <Block key={block.key} block={block} isEmbedded={isEmbedded} />
        ))}
    </>
  );
}

function Block({ block, isEmbedded }: { block: BlockData; isEmbedded: boolean }) {
  const targetIdx = useAtomValue(targetIdxAtom);
  const blockModifier = useAtomValue(useMemo(() => createBlockModifierAtom(block), [block]));
  const setHoveredBlock = useSetAtom(hoveredBlockAtom);
  const toggleSelection = useSetAtom(toggleSelectionAtom);

  const classes = classNames({
    block: true,
    hovered: blockModifier.isHovered,
    selected: blockModifier.isSelected,
    emphasized: blockModifier.isEmphasized,
  });

  // Offset is a positive number relative to last token in sample text
  const gridColumn = targetIdx - block.tokenOffset + 1;
  // Row 1 is the sample text, Row 2 is the 0th layerIdx
  const gridRow = block.layerIdx + 2;
  return (
    <div className="cell" style={{ gridColumn: gridColumn, gridRow: gridRow }}>
      <div
        className={classes}
        onMouseEnter={() => setHoveredBlock(block)}
        onMouseLeave={() => setHoveredBlock(null)}
        onClick={(e: React.MouseEvent) => {
          // Toggle block selection.
          toggleSelection(blockModifier.isSelected ? null : block);

          // Prevent event from bubbling up to parent elements
          e.stopPropagation();
        }}
      >
        {Object.values(block.features)
          .sort((a, b) => a.featureId - b.featureId)
          .map((feature) => (
            <Feature key={feature.key} feature={feature} isEmbedded={isEmbedded} />
          ))}
      </div>
    </div>
  );
}

function Feature({ feature, isEmbedded }: { feature: BlockFeatureData; isEmbedded: boolean }) {
  const modelId = useAtomValue(modelIdAtom);
  const sampleId = useAtomValue(sampleIdAtom);
  const setHoveredFeature = useSetAtom(hoveredFeatureAtom);
  const toggleSelection = useSetAtom(toggleSelectionAtom);
  const featureModifier = useAtomValue(
    useMemo(() => createFeatureModifierAtom(feature), [feature])
  );

  const classes = classNames({
    feature: true,
    highlighted: featureModifier.isHovered,
    selected: featureModifier.isSelected,
    related: featureModifier.isRelatedToFocused,
    active: featureModifier.isActive,
    "fill-gray": featureModifier.isGray,
    [`fill-weight-${featureModifier.fillWeight}`]: featureModifier.isActive,
    [`text-weight-${featureModifier.textWeight}`]: true,
    [`text-color-${featureModifier.textColor}`]: true,
  });

  const onClick = (e: React.MouseEvent) => {
    if (!isEmbedded) {
      // Select the feature.
      toggleSelection(featureModifier.isSelected ? null : feature);

      // Prevent event from bubbling up to parent elements.
      e.stopPropagation();
    } else {
      // Open the feature in a new tab.
      const sampleUrl = `#${getInspectSamplePath(modelId, sampleId, feature.key)}`;
      window.open(sampleUrl, "child");
    }
  };

  return (
    <span
      id={feature.elementId}
      className={classes}
      onMouseEnter={() => setHoveredFeature(feature)}
      onMouseLeave={() => setHoveredFeature(null)}
      onClick={onClick}
    >
      {feature.featureId}
      {featureModifier.isNeighborFocused && <FeatureProfilePreloader feature={feature} />}
    </span>
  );
}

/**
 * Preloads the feature profile for the given feature. Doesn't render anything.
 */
function FeatureProfilePreloader({ feature }: { feature: BlockFeatureData }) {
  useAtom(useMemo(() => createFeatureProfileAtom(feature), [feature]));
  return <></>;
}

function Connections() {
  const connections = useAtomValue(connectionsAtom);

  return (
    <>
      {Object.values(connections).map((connection) => (
        <Connection connection={connection} key={connection.key} />
      ))}
    </>
  );
}

function Connection({ connection }: { connection: ConnectionData }) {
  const targetIdx = useAtomValue(targetIdxAtom);
  const connectionModifier = useAtomValue(
    useMemo(() => createConnectionModifierAtom(connection), [connection])
  );

  // Offset is a positive number relative to last token in sample text
  const gridColumn = targetIdx - connection.upstreamTokenOffset + 1;
  // Row 1 is the sample text, Row 2 is the 0th layerIdx
  const gridRow = connection.upstreamLayerIdx + 2;
  // How many columns the connection spans
  const span = connection.upstreamTokenOffset - connection.downstreamTokenOffset + 1;
  // Offset connections by a few pixels based on downstream token offset to reduce overlap
  const displayOffset = 4 - (connection.downstreamTokenOffset % 9);
  const displayOffsetDirection = displayOffset > 0 ? "r" + displayOffset : "l" + displayOffset * -1;

  const classes = classNames({
    connection: true,
    gray: connectionModifier.isGray,
    [`width-${connectionModifier.width}`]: true,
    [`weight-${connectionModifier.weight}`]: true,
    [`span-${span}`]: true,
    [`offset-${displayOffsetDirection}`]: true,
  });

  return (
    <div className={classes} style={{ gridColumn: gridColumn, gridRow: gridRow }}>
      <div className="line">
        <div className="segment upstream"></div>
        <div className="segment downstream"></div>
      </div>
    </div>
  );
}

export { AblationMap };
