import classNames from "classnames";
import { useAtom, useAtomValue, useSetAtom } from "jotai";
import { useMemo, useState } from "react";
import { FaChevronDown, FaLayerGroup } from "react-icons/fa6";

import { BlockData, BlockFeatureData, blocksAtom } from "../../../stores/Block";
import { createFeatureProfileAtom, FeatureProfile } from "../../../stores/Feature";
import { printableTokensAtom, targetIdxAtom } from "../../../stores/Graph";
import { SampleData } from "../../../stores/Sample";
import { SamplingStrategies, samplingStrategyAtom } from "../../../stores/Search";
import {
  hoveredUpstreamOffsetAtom,
  selectionStateAtom,
  toggleSelectionAtom,
} from "../../../stores/Selection";
import { CloseButton } from "./close";
import { ErrorMessage, LoadingMessage } from "./loading";
import { SearchableSamples } from "./samples";

function FeatureSidebar({ feature }: { feature: BlockFeatureData }) {
  const [{ data: featureProfile, isPending, isError }] = useAtom(
    useMemo(() => createFeatureProfileAtom(feature), [feature])
  );
  const samplingStrategy = useAtomValue(samplingStrategyAtom);

  if (isPending) {
    return (
      <>
        <FeatureSidebarHeader feature={feature} />
        <LoadingMessage />
      </>
    );
  }

  if (isError || !featureProfile) {
    return (
      <>
        <FeatureSidebarHeader feature={feature} />
        <ErrorMessage />
      </>
    );
  }

  let samples: SampleData[];
  switch (samplingStrategy) {
    case SamplingStrategies.Cluster:
      // Show all samples if using the clustering strategy
      samples = featureProfile.samples;
      break;
    case SamplingStrategies.Similar:
      // Show samples with the closest activations
      samples = featureProfile.samples
        .sort((a, b) => {
          const aActivation = a.tokens[a.targetIdx].activation;
          const bActivation = b.tokens[b.targetIdx].activation;
          return (
            Math.abs(aActivation - feature.activation) - Math.abs(bActivation - feature.activation)
          );
        })
        .slice(0, 25);
      break;
    case SamplingStrategies.Top:
      // Show samples with the top 25 activations
      samples = featureProfile.samples
        .sort((a, b) => b.tokens[b.targetIdx].activation - a.tokens[a.targetIdx].activation)
        .slice(0, 25);
  }

  return (
    <>
      <FeatureSidebarHeader feature={feature} />
      <FeatureActivationsSection feature={feature} featureProfile={featureProfile} />
      <UpstreamAblationsSection feature={feature} featureProfile={featureProfile} />
      <FeatureSidebarSamplesHeader />
      <SearchableSamples
        samples={samples}
        feature={feature}
        activationHistogram={featureProfile.activationHistogram}
        titleComponent={<FeatureSidebarSamplingStrategy />}
      />
    </>
  );
}

function FeatureSidebarSamplingStrategy() {
  const [isMenuOpen, setIsMenuOpen] = useState(false);
  const [strategy, setStrategy] = useAtom(samplingStrategyAtom);

  return (
    <div
      className="feature-sampling-strategy"
      onClick={() => setIsMenuOpen(!isMenuOpen)}
      onBlur={() => setIsMenuOpen(false)}
      tabIndex={0}
    >
      <span className="selected-option">
        <span>
          {strategy === SamplingStrategies.Cluster ? "From Cluster" : null}
          {strategy === SamplingStrategies.Similar ? "Similar Activations" : null}
          {strategy === SamplingStrategies.Top ? "Top Activations" : null}
        </span>
        <FaChevronDown className="icon" />
      </span>

      {isMenuOpen && (
        <ul className="menu">
          <li className="header">Sampling Strategy</li>
          <li className="option" onClick={() => setStrategy(SamplingStrategies.Cluster)}>
            <h4>From Cluster</h4>
            <p>
              Cluster dataset samples using circuit features and boost the importance of the
              selected feature.
            </p>
          </li>
          <li className="option" onClick={() => setStrategy(SamplingStrategies.Similar)}>
            <h4>Similar Activations</h4>
            <p>Show dataset samples for tokens with similar activation values.</p>
          </li>
          <li className="option" onClick={() => setStrategy(SamplingStrategies.Top)}>
            <h4>Top Activations</h4>
            <p>Show dataset samples for tokens with 90th percentile activations.</p>
          </li>
        </ul>
      )}
    </div>
  );
}

function FeatureSidebarHeader({ feature }: { feature: BlockFeatureData }) {
  return (
    <header>
      <span className="feature-location">
        <span>Feature #{feature.featureId}</span>
        <span className="layer">
          <FaLayerGroup className="icon" />
          {feature.layerIdx > 0 ? `${feature.layerIdx}` : "Embedding"}
        </span>
      </span>
      <CloseButton />
    </header>
  );
}

function FeatureSidebarSamplesHeader() {
  return <header className="feature-samples-header">Examples</header>;
}

function FeatureActivationsSection({
  feature,
  featureProfile,
}: {
  feature: BlockFeatureData;
  featureProfile: FeatureProfile;
}) {
  // Activation values to use for the stacked bar chart
  const maxActivation = featureProfile.maxActivation;
  const featureActivation = feature.activation;
  const activationWidth = featureActivation / maxActivation;
  const backgroundWidth = 1 - activationWidth;

  return (
    <section className="activations">
      <h3>Activation</h3>
      <p>
        <span>
          <span>
            {featureActivation.toFixed(3)} / {maxActivation.toFixed(3)}
          </span>
          <span className="percentage">
            ({((featureActivation / maxActivation) * 100).toFixed(1)}%)
          </span>
        </span>
      </p>
      <table className="charts-css bar multiple stacked">
        <tbody>
          <tr>
            <td style={{ "--size": activationWidth } as React.CSSProperties}></td>
            <td style={{ "--size": backgroundWidth } as React.CSSProperties}></td>
          </tr>
        </tbody>
      </table>
    </section>
  );
}

function UpstreamAblationsSection({
  feature,
  featureProfile,
}: {
  feature: BlockFeatureData;
  featureProfile: FeatureProfile;
}) {
  const selectionState = useAtomValue(selectionStateAtom);
  const printableTokens = useAtomValue(printableTokensAtom);
  const targetIdx = useAtomValue(targetIdxAtom);
  const setHoveredUpstreamOffset = useSetAtom(hoveredUpstreamOffsetAtom);

  // For selecting blocks
  const blocks = useAtomValue(blocksAtom);
  const toggleSelection = useSetAtom(toggleSelectionAtom);

  // Construct a list of (token, offset) pairs for upstream ablations.
  const upstreamOffsets = [
    ...new Set(Object.values(feature.ablatedBy).map((a) => a.tokenOffset)),
  ].sort((a, b) => a - b);
  const upstreamAblations = upstreamOffsets.map((offset) => {
    const token = printableTokens[targetIdx - offset];
    return [token, offset] as [string, number];
  });

  // Are there any upstream ablations?
  if (upstreamAblations.length === 0) {
    return <></>;
  }

  // Compute ideal width for chart labels (in ch units)
  const chartLabelSize = Math.max(...upstreamAblations.map(([token]) => token.length)) + 2;

  // Compute the maximum ablation value for the chart
  const maxAblation = Math.max(
    ...upstreamAblations.map(([, offset]) => feature.groupAblations[offset])
  );

  return (
    <section className="ablations">
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
                toggleSelection(blocks[BlockData.getKey(offset, feature.layerIdx - 1)])
              }
            >
              <th scope="row">
                <pre className="token">{token}</pre>
              </th>
              <td
                style={
                  {
                    "--size": feature.groupAblations[offset] / maxAblation,
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

export { FeatureSidebar };
