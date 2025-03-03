import { ArchitectureDiagram } from "./ArchitectureDiagram";

function FeatureExtraction() {
  return (
    <section id="FeatureExtraction">
      <h2>Feature Extraction</h2>
      <p>
        In the field of mechanistic interpretability, features represent the internal “building
        blocks” that a model uses to produce predictions. Ideal features are “interpretable” and
        “sparsely activated”. One may naively expect that the neurons of a neural network would
        produce such features; however, neural activations are noisy and yield an opaque
        understanding of what’s happening inside a model.
      </p>
      <p>
        Lately, interpretability researchers have been using “
        <a
          href="https://adamkarvonen.github.io/machine_learning/2024/06/11/sae-intuitions.html"
          target="_blank"
          rel="noreferrer"
        >
          sparse autoencoders
        </a>
        ” to extract sets of interpretable features from individual layers of activity, which is
        what this app uses as a prerequisite to circuit extraction. This app specifically use a “
        <a href="https://arxiv.org/pdf/2407.14435" target="_blank" rel="noreferrer">
          JumpReLU sparse autoencoder
        </a>
        ” (a.k.a JumpReLU SAE) to produce features representing the activations of the residual
        stream before and after every transformer block in an LLM.
      </p>
      <ArchitectureDiagram />
    </section>
  );
}

export { FeatureExtraction };
