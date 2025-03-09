import { useAtom, useAtomValue } from "jotai";

import classNames from "classnames";
import { useState } from "react";
import { FaChevronDown, FaLayerGroup } from "react-icons/fa6";
import { probabilitiesAtom, targetTokenAtom } from "../../../stores/Graph";
import { layerProfilesAtom } from "../../../stores/Layer";
import { predictionDataAtom } from "../../../stores/Prediction";
import { selectedLayerIdxAtom } from "../../../stores/Selection";
import { ProbabilitiesTable } from "../ProbabilitiesTable";
import { ErrorMessage, LoadingMessage } from "./loading";
import { SearchableSamples } from "./samples";

function PredictionSidebar() {
  return (
    <>
      <PredictionHeader />
      <LLMPrediction />
      <CircuitPrediction />
      <KldSection />
      <PredictionSamplesSection />
    </>
  );
}

function PredictionHeader() {
  const targetToken = useAtomValue(targetTokenAtom);

  return (
    <header>
      <span>
        Next Token After <pre className="token">{targetToken}</pre>
      </span>
    </header>
  );
}

function LLMPrediction() {
  const probabilities = useAtomValue(probabilitiesAtom);
  if (!probabilities || Object.keys(probabilities).length === 0) {
    return <></>;
  }

  return (
    <section className="circuit-prediction">
      <h3>LLM Prediction</h3>
      <ProbabilitiesTable probabilities={probabilities} />
    </section>
  );
}

function CircuitPrediction() {
  const layerProfiles = useAtomValue(layerProfilesAtom);
  const [isMenuOpen, setIsMenuOpen] = useState(false);
  const [selectedLayerIdx, setSelectedLayerIdx] = useAtom(selectedLayerIdxAtom);
  const focusedLayerIdx = selectedLayerIdx === null ? layerProfiles.length - 1 : selectedLayerIdx;
  if (!layerProfiles || layerProfiles.length === 0) {
    return <></>;
  }

  const probabilities = layerProfiles[focusedLayerIdx]?.probabilities || {};

  return (
    <section className="circuit-prediction">
      <h3
        className="layer-location"
        onClick={() => setIsMenuOpen(!isMenuOpen)}
        onBlur={() => setIsMenuOpen(false)}
        tabIndex={0}
      >
        Prediction Using{" "}
        <span className="layer">
          <FaLayerGroup className="icon" />
          {focusedLayerIdx > 0 ? `${focusedLayerIdx}` : "Embedding"}
          <FaChevronDown className="dropdown-icon" />
        </span>
        {isMenuOpen && (
          <ul className="menu">
            <li className="header">Layers</li>
            {layerProfiles.map((layerProfile, i) => (
              <li key={i} className="option" onClick={() => setSelectedLayerIdx(i)}>
                <h4>
                  {i > 0 ? `Layer ${i}` : "Embedding"} &ndash; {layerProfile.kld.toFixed(2)}
                </h4>
              </li>
            ))}
          </ul>
        )}
      </h3>
      <ProbabilitiesTable probabilities={probabilities} />
    </section>
  );
}

function KldSection() {
  const layerProfiles = useAtomValue(layerProfilesAtom);
  if (!layerProfiles || layerProfiles.length === 0) {
    return <></>;
  }

  const maxKld = Math.max(...layerProfiles.map((layerProfile) => layerProfile.kld));

  return (
    <>
      <header>KL Divergence by Layer</header>
      <section className="klds">
        <table className="charts-css column show-primary-axis show-labels data-spacing-1">
          <tbody>
            {layerProfiles.map((layerProfile, i) => (
              <tr key={i}>
                <th>{layerProfile.idx}</th>
                <td
                  style={
                    {
                      "--size": layerProfile.kld / maxKld,
                    } as React.CSSProperties
                  }
                  className={classNames({
                    low: layerProfile.kld > 0.0 && layerProfile.kld <= 0.25,
                    medium: layerProfile.kld > 0.25 && layerProfile.kld <= 0.5,
                    high: layerProfile.kld > 0.5,
                  })}
                >
                  {layerProfile.kld / maxKld > 0.5 ? layerProfile.kld.toFixed(2) : null}
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      </section>
    </>
  );
}

function PredictionSamplesSection() {
  const [{ data: predictionData, isPending, isError }] = useAtom(predictionDataAtom);

  if (isPending) {
    return (
      <>
        <header>
          <PredictionSamplesTitle />
        </header>
        <LoadingMessage />
      </>
    );
  }

  if (isError || !predictionData) {
    return (
      <>
        <header>
          <PredictionSamplesTitle />
        </header>
        <ErrorMessage />
      </>
    );
  }

  return (
    <SearchableSamples
      samples={predictionData.samples}
      titleComponent={<PredictionSamplesTitle />}
    />
  );
}

function PredictionSamplesTitle() {
  return <span>Similar Tokens</span>;
}

export { PredictionSidebar };
