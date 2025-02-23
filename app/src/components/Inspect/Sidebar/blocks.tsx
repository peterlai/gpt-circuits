import { useAtom, useAtomValue, useSetAtom } from "jotai";
import { useMemo } from "react";
import { FaLayerGroup } from "react-icons/fa6";
import { BlockData, createBlockProfileAtom } from "../../../stores/Block";
import { printableTokensAtom, targetIdxAtom } from "../../../stores/Graph";
import { toggleSelectionAtom } from "../../../stores/Selection";
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
  const toggleSelection = useSetAtom(toggleSelectionAtom);

  return (
    <section className="block-profile">
      <h3>Features</h3>
      <table className="charts-css bar multiple stacked show-labels data-spacing-1">
        <tbody>
          {Object.values(block.features).map((feature) => {
            return (
              <tr key={feature.featureId}>
                <th>
                  <span
                    onClick={() => {
                      // Select feature if clicked
                      toggleSelection(feature);
                    }}
                  >
                    #{feature.featureId}
                  </span>
                </th>
                <td style={{ "--size": feature.normalizedActivation } as React.CSSProperties}>
                  <span className="data"> {feature.activation.toFixed(2)}</span>
                </td>
                <td
                  style={{ "--size": 1.0 - feature.normalizedActivation } as React.CSSProperties}
                ></td>
              </tr>
            );
          })}
        </tbody>
      </table>
    </section>
  );
}

function BlockSamplesTitle() {
  return <span>Similar Examples</span>;
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
