import classNames from "classnames";
import { useAtom, useAtomValue, useSetAtom } from "jotai";
import { useMemo } from "react";
import { FaLayerGroup } from "react-icons/fa6";
import { MdClose } from "react-icons/md";

import { BlockFeatureData } from "../../../stores/Block";
import { createFeatureProfileAtom, FeatureProfile } from "../../../stores/Feature";
import { printableTokensAtom, targetIdxAtom } from "../../../stores/Graph";
import { isMobile, isSidebarOpenAtom } from "../../../stores/Navigation";
import {
  hoveredUpstreamOffsetAtom,
  selectionStateAtom,
  toggleSelectionAtom,
} from "../../../stores/Selection";
import { ErrorMessage, LoadingMessage } from "./loading";
import { SearchableSamples } from "./samples";

function FeatureSidebar({ feature }: { feature: BlockFeatureData }) {
  const [{ data: featureProfile, isPending, isError }] = useAtom(
    useMemo(() => createFeatureProfileAtom(feature), [feature])
  );

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

  return (
    <>
      <FeatureSidebarHeader feature={feature} />
      <FeatureActivationsSection feature={feature} featureProfile={featureProfile} />
      <UpstreamAblationsSection feature={feature} featureProfile={featureProfile} />
      <SearchableSamples
        samples={featureProfile.samples}
        feature={feature}
        activationHistogram={featureProfile.activationHistogram}
      />
    </>
  );
}

function FeatureSidebarHeader({ feature }: { feature: BlockFeatureData }) {
  const setIsSidebarOpen = useSetAtom(isSidebarOpenAtom);
  const toggleSelection = useSetAtom(toggleSelectionAtom);

  const closeHandler = () => {
    if (isMobile()) {
      // Wait for animation to finish before clearing the selection.
      setIsSidebarOpen(false);
      setTimeout(() => {
        toggleSelection(null);
      }, 300);
    } else {
      toggleSelection(null);
    }
  };

  return (
    <header>
      <span className="feature-location">
        <span>Feature #{feature.featureId}</span>
        <span className="layer">
          <FaLayerGroup className="icon" />
          {feature.layerIdx > 0 ? `${feature.layerIdx}` : "Embedding"}
        </span>
      </span>
      <MdClose className="close" onClick={closeHandler} />
    </header>
  );
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
            >
              <th scope="row">
                <pre className="token">{token}</pre>
              </th>
              <td
                style={
                  {
                    "--size": feature.groupAblations[offset] / featureProfile.maxActivation,
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
