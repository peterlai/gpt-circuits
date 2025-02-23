import { useAtom, useAtomValue } from "jotai";

import {
  circuitProbabilitiesAtom,
  probabilitiesAtom,
  targetTokenAtom,
} from "../../../stores/Graph";
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
  const probabilities = useAtomValue(circuitProbabilitiesAtom);
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
  return <span>Similar Examples</span>;
}

export { PredictionSidebar };
