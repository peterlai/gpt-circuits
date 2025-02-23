import { useAtom, useAtomValue } from "jotai";
import { FaLayerGroup } from "react-icons/fa6";
import { BlockData } from "../../../stores/Block";
import { printableTokensAtom, targetIdxAtom } from "../../../stores/Graph";
import { predictionDataAtom } from "../../../stores/Prediction";
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
  return (
    <section className="block-profile">
      <h3>Features</h3>
      <table className="charts-css bar multiple stacked show-labels data-spacing-1">
        <tbody>
          {Object.values(block.features).map((feature) => {
            return (
              <tr key={feature.featureId}>
                <th scope="row">
                  <span>#{feature.featureId}</span>
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
  // TODO: Switch to block-specific samples
  const [{ data: predictionData, isPending, isError }] = useAtom(predictionDataAtom);

  if (isPending) {
    return <LoadingMessage />;
  }

  if (isError || !predictionData) {
    return <ErrorMessage />;
  }

  return (
    <SearchableSamples samples={predictionData.samples} titleComponent={<BlockSamplesTitle />} />
  );
}

export { BlockSidebar };
