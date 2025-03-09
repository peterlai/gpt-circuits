import { useAtom, useAtomValue } from "jotai";

import { probabilitiesAtom, targetTokenAtom } from "../../../stores/Graph";
import { layerProfilesAtom } from "../../../stores/Layer";
import { predictionDataAtom } from "../../../stores/Prediction";
import { ProbabilitiesTable } from "../ProbabilitiesTable";
import { ErrorMessage, LoadingMessage } from "./loading";
import { SearchableSamples } from "./samples";

function PredictionSidebar() {
  return (
    <>
      <PredictionHeader />
      <LLMPrediction />
      <CircuitPrediction />
      <PredictionSamplesSection />
    </>
  );
}

function PredictionHeader() {
  const targetToken = useAtomValue(targetTokenAtom);

  return (
    <header>
      <span>
        Next token after <pre className="token">{targetToken}</pre>
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
  const probabilities =
    layerProfiles.length > 0 ? layerProfiles[layerProfiles.length - 1].probabilities : null;
  if (!probabilities || Object.keys(probabilities).length === 0) {
    return <></>;
  }

  return (
    <section className="circuit-prediction">
      <h3>Circuit Prediction</h3>
      <ProbabilitiesTable probabilities={probabilities} />
    </section>
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
