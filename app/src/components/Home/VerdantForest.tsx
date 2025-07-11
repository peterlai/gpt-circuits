import { SampleData } from "../../stores/Sample";
import { AlignmentOptions } from "../../stores/Search";
import { getInspectSamplePath } from "../../views/App/urls";
import { SamplesList } from "../SamplesList";
import { similarSamples } from "./SimilarSamples";
import "./VerdantForest.scss";

function VerdantForest() {
  const modelId = "toy-stories";
  const sampleId = "2747096.11";
  const version = "0.25";
  const selectionKey = "0.4";
  const samplePath = `#${getInspectSamplePath(modelId, sampleId, version)}`;
  const samplePathWithSelectionKey = `#${getInspectSamplePath(
    modelId,
    sampleId,
    version,
    selectionKey
  )}`;

  let samplesList: SampleData[] = similarSamples.map((sample) => {
    // Create arrays of activations and normalized activations from tokens
    const activations = sample.decodedTokens.map(() => 0); // Default all to 0
    const normalizedActivations = sample.decodedTokens.map(() => 0); // Default all to 0

    // If sample has tokens with activation data, use them
    if (sample.tokens) {
      sample.tokens.forEach((token, idx) => {
        if (idx < activations.length) {
          activations[idx] = token.activation || 0;
          normalizedActivations[idx] = token.normalizedActivation || 0;
        }
      });
    }

    return new SampleData(
      sample.text,
      sample.decodedTokens,
      activations,
      normalizedActivations,
      sample.targetIdx,
      sample.absoluteTokenIdx,
      sample.modelId
    );
  });

  return (
    <section id="VerdantForest">
      <h2>“A fluffy blue creature roamed the verdant forest.”</h2>
      <p>
        About a year ago, I watched a 3Blue1Brown{" "}
        <a
          href="https://www.youtube.com/watch?v=eMlx5fFNoYc"
          target="_blank"
          rel="noopener noreferrer"
        >
          video
        </a>{" "}
        on how a model’s self-attention mechanism is used to predict the next token in a sequence,
        and I was surprised by how little we know of what actually happens when processing the
        sentence, “A fluffy blue creature roamed the verdant forest.” The following illustration
        from this video was produced through use of hypothetical model representations.
      </p>
      <figure>
        <figcaption>
          <a
            href="https://youtu.be/eMlx5fFNoYc?si=kXz3zyvrdnUC_mUs&t=367"
            target="_blank"
            rel="noopener noreferrer"
          >
            What Might Happen
          </a>
        </figcaption>
        <a
          href="https://youtu.be/eMlx5fFNoYc?si=kXz3zyvrdnUC_mUs&t=367"
          target="_blank"
          rel="noopener noreferrer"
        >
          <img
            src={`${process.env.PUBLIC_URL}/home/hypothetical.png`}
            alt="Circuit visualization"
          />
        </a>
      </figure>
      <h3>Mechanistic Interpretability</h3>
      <p>
        Since then, the field of{" "}
        <a
          href="https://www.transformer-circuits.pub/2022/mech-interp-essay"
          target="_blank"
          rel="noopener noreferrer"
        >
          mechanistic interpretability
        </a>{" "}
        has seen significant advancements, and we are now better able to “decompose” models into
        interpretable circuits that help explain how LLMs produce predictions. For this sentence
        specifically, our “debugger” produces a few high-level insights. We can see that the model
        is capable of identifying a “creature” as the subject of this sentence. Because the last
        token is a period, the model sees that it needs to start a new sentence, and it does so by
        borrowing mostly from sentences following the description of animals.
      </p>
      <figure>
        <figcaption>
          <a href={samplePath} target="_blank" rel="noopener noreferrer">
            What Does Happen
          </a>
        </figcaption>
        <a href={samplePath} target="_blank" rel="noopener noreferrer">
          <img
            src={`${process.env.PUBLIC_URL}/home/verdant-forest.png`}
            alt="Circuit for verdant forest sentence"
          />
        </a>
        <p>
          The model identifies a "creature" as the subject of this sentence and seems to use this
          information to form the start of a new sentence.
        </p>
      </figure>
      <figure>
        <figcaption>
          <a href={samplePathWithSelectionKey} target="_blank" rel="noopener noreferrer">
            Similar Sequences from Training Data
          </a>
        </figcaption>
        <SamplesList
          samples={samplesList}
          alignment={AlignmentOptions.Token}
          rightPadding={20}
          hideActivation={true}
          showIcon={true}
        />
        <p>
          These sequences from the model's training dataset produce feature activations in the last
          layer that most strongly resemble the feature activations produced by the original
          sentence.
        </p>
      </figure>
    </section>
  );
}

export { VerdantForest };
