import { getInspectSamplePath } from "../../views/App/urls";
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
      </figure>
      <figure>
        <figcaption>
          <a href={samplePathWithSelectionKey} target="_blank" rel="noopener noreferrer">
            Similar Sequences from Training Data
          </a>
        </figcaption>
        <a href={samplePathWithSelectionKey} target="_blank" rel="noopener noreferrer">
          <img
            src={`${process.env.PUBLIC_URL}/home/similar-sequences.png`}
            alt="Similar sequences from training data"
          />
        </a>
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
