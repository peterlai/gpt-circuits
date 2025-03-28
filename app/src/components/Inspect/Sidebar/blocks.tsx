import classNames from "classnames";
import { useAtom, useAtomValue, useSetAtom } from "jotai";
import { useMemo } from "react";
import { FaLayerGroup } from "react-icons/fa6";
import { BlockData, blocksAtom, createBlockProfileAtom } from "../../../stores/Block";
import { printableTokensAtom, targetIdxAtom } from "../../../stores/Graph";
import {
  hoveredUpstreamOffsetAtom,
  selectionStateAtom,
  toggleSelectionAtom,
} from "../../../stores/Selection";
import { CloseButton } from "./close";
import { ErrorMessage, LoadingMessage } from "./loading";
import { SearchableSamples } from "./samples";

function BlockSidebar({ block }: { block: BlockData }) {
  return (
    <>
      <BlockHeader block={block} />
      <BlockProfile block={block} />
      <BlockSamplesSection block={block} />
    </>
  );
}

function BlockHeader({ block }: { block: BlockData }) {
  const printableTokens = useAtomValue(printableTokensAtom);
  const targetIdx = useAtomValue(targetIdxAtom);
  const blockToken = printableTokens[targetIdx - block.tokenOffset];

  return (
    <header>
      <span className="feature-location">
        <pre className="token">{blockToken}</pre>
        <span className="layer">
          <FaLayerGroup className="icon" />
          {block.layerIdx > 0 ? `${block.layerIdx}` : "Embedding"}
        </span>
      </span>
      <CloseButton />
    </header>
  );
}

function BlockProfile({ block }: { block: BlockData }) {
  // For selecting blocks
  const blocks = useAtomValue(blocksAtom);
  const selectionState = useAtomValue(selectionStateAtom);
  const toggleSelection = useSetAtom(toggleSelectionAtom);
  const setHoveredUpstreamOffset = useSetAtom(hoveredUpstreamOffsetAtom);

  // All ablations with this block as a downstream destination
  const allAblations = Object.values(block.features).flatMap((f) =>
    Object.values(f.ablatedBy).map((a) => ({ ...a, feature: f }))
  );

  // All upstream offsets for this block
  const upstreamOffsets = [...new Set(Object.values(allAblations).map((a) => a.tokenOffset))].sort(
    (a, b) => a - b
  );

  // Construct a list of (token, offset) pairs for upstream ablations.
  const targetIdx = useAtomValue(targetIdxAtom);
  const printableTokens = useAtomValue(printableTokensAtom);
  const upstreamAblations = upstreamOffsets.map((offset) => {
    const token = printableTokens[targetIdx - offset];
    return [token, offset] as [string, number];
  });

  // Map offsets to token importance
  const tokenImportances = {} as Record<number, number>;
  upstreamOffsets.forEach((offset) => {
    tokenImportances[offset] = Math.max(0.001, block.upstreamImportances[offset]);
  });

  // Compute ideal width for chart labels (in ch units)
  const chartLabelSize = Math.max(...upstreamAblations.map(([token]) => token.length)) + 2;

  return (
    <section className="block-profile">
      <h3>Upstream Tokens</h3>
      <table className="charts-css bar show-labels data-spacing-1">
        <tbody style={{ "--labels-size": `${chartLabelSize}ch` } as React.CSSProperties}>
          {upstreamAblations.map(([token, offset]) => (
            <tr
              key={offset}
              className={classNames({
                ablation: true,
                hovered: offset === selectionState.hoveredUpstreamOffset,
              })}
              onMouseEnter={() => setHoveredUpstreamOffset(offset)}
              onMouseLeave={() => setHoveredUpstreamOffset(null)}
              onClick={() =>
                // Select upstream block
                toggleSelection(blocks[BlockData.getKey(offset, block.layerIdx - 1)])
              }
            >
              <th scope="row">
                <pre className="token">{token}</pre>
              </th>
              <td
                style={
                  {
                    "--size": tokenImportances[offset],
                  } as React.CSSProperties
                }
              ></td>
            </tr>
          ))}
        </tbody>
      </table>
    </section>
  );
}

function BlockSamplesTitle() {
  return <span>Similar Tokens</span>;
}

function BlockSamplesSection({ block }: { block: BlockData }) {
  const [{ data: blockProfile, isPending, isError }] = useAtom(
    useMemo(() => createBlockProfileAtom(block), [block])
  );

  if (isPending) {
    return <LoadingMessage />;
  }

  if (isError || !blockProfile) {
    return <ErrorMessage />;
  }

  return (
    <SearchableSamples samples={blockProfile.samples} titleComponent={<BlockSamplesTitle />} />
  );
}

export { BlockSidebar };
