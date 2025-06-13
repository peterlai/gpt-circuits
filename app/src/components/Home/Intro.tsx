import { useAtom } from "jotai";
import { useEffect, useRef } from "react";
import { FaCalendarDays, FaUpRightFromSquare, FaUser } from "react-icons/fa6";
import { Tooltip, TooltipRefProps } from "react-tooltip";

import { isMenuOpenAtom, isMobile } from "../../stores/Navigation";
import { getInspectSamplePath } from "../../views/App/urls";
import { SampleAnchor } from "../Home/Sample";

import "./Intro.scss";

function Intro() {
  const modelId = "toy-v0";
  const sampleId = "val.0.69248.76";
  const version = "0.1";
  const samplePath = `#${getInspectSamplePath(modelId, sampleId, version)}`;
  const [isMenuOpen, setIsMenuOpen] = useAtom(isMenuOpenAtom);
  const tooltipRef = useRef<TooltipRefProps>(null);
  const tooltipPosition = isMobile() ? undefined : { x: 275, y: 300 };

  // Show the menu before showing the app tooltip.
  const openAppHandler = (e: React.MouseEvent) => {
    const delay = isMenuOpen ? 0 : 300;
    setIsMenuOpen(true);
    setTimeout(() => {
      tooltipRef.current?.open();
    }, delay);
    e.preventDefault();
  };

  // Auto-close the app tooltip after a delay.
  const tooltipShownHandler = () => {
    setTimeout(() => {
      tooltipRef.current?.close();
    }, 8000);
  };

  // Close the app tooltip when the menu is closed.
  useEffect(() => {
    if (!isMenuOpen) {
      tooltipRef.current?.close();
    }
  }, [isMenuOpen]);

  return (
    <section id="Intro">
      <h1>
        <img src={`${process.env.PUBLIC_URL}/logo.png`} alt="logo" />
        <span>Proof-of-Concept LLM Debugger</span>
      </h1>
      <div className="header-meta">
        <div className="authors">
          <div className="author">
            <FaUser className="icon" />
            <span>Peter Lai</span>
          </div>
        </div>
        <div className="publication-date">
          <FaCalendarDays className="icon" />
          <span>June 12, 2025</span>
        </div>
      </div>
      <p>
        There are several great{" "}
        <a href="https://www.3blue1brown.com/lessons/gpt" target="_blank" rel="noopener noreferrer">
          tutorials
        </a>{" "}
        on how LLMs process information; however, these tutorials use hypothetical examples to
        visualize the flow of data. What actually happens after feeding a simple sentence through an
        LLM? This{" "}
        <a href="#/" onClick={openAppHandler}>
          app
        </a>{" "}
        implements a proof-of-concept LLM “debugger” that uses real internal representations of
        learned concepts to construct an explanation of how simple LLMs produce simple behaviors.
      </p>
      <figure>
        <figcaption>
          <SampleAnchor
            modelId={modelId}
            sampleId={sampleId}
            version={version}
            text="Predicting What Comes After the Letter “s”"
          />
        </figcaption>
        <a href={samplePath} target="_blank" rel="noopener noreferrer">
          <img
            src={`${process.env.PUBLIC_URL}/home/thinks-circuit.png`}
            alt="Circuit visualization"
          />
        </a>
      </figure>
      <p>
        This app automates “circuit” visualizations for two tiny GPT-2 models – one is
        character-based and has been trained using text from{" "}
        <a
          href="https://huggingface.co/datasets/karpathy/tiny_shakespeare"
          target="_blank"
          rel="noopener noreferrer"
        >
          Shakespeare plays
        </a>
        . A second uses a more conventional GPT-2 tokenizer and has been trained using{" "}
        <a
          href="https://huggingface.co/datasets/roneneldan/TinyStories"
          target="_blank"
          rel="noopener noreferrer"
        >
          short stories
        </a>{" "}
        for children. For a more detailed explanation of what “circuits” are and how they're
        extracted,{" "}
        <a
          href="https://www.lesswrong.com/posts/rxR5p9Qha937wTTBp/proof-of-concept-debugger-for-a-small-llm"
          target="_blank"
          rel="noopener noreferrer"
        >
          this post
        </a>{" "}
        offers a deep dive into the heuristics used to assemble graphs and extract internal
        representations.
      </p>
      <p className="deepdive-link">
        <a
          href="https://www.lesswrong.com/posts/rxR5p9Qha937wTTBp/proof-of-concept-debugger-for-a-small-llm"
          target="_blank"
          rel="noopener noreferrer"
          className="external-link"
        >
          Circuit Extraction Details
          <FaUpRightFromSquare className="external-icon" />
        </a>
      </p>
      <p>
        Use the{" "}
        <a href="#/" onClick={openAppHandler}>
          menu
        </a>{" "}
        on the left to browse example circuits. You'll find that each visual block is composed of “
        <a
          href="https://transformer-circuits.pub/2023/monosemantic-features/index.html"
          target="_blank"
          rel="noopener noreferrer"
        >
          features
        </a>
        ” that fire upon encountering specific sequences of tokens. These features act in ensemble
        to produce higher level features, which then activate in response to (i) longer sequences
        and (ii) specific grammatical patterns. What results is a graph tying the effects of
        specific input tokens to the probabilities of the next token in a sequence.
      </p>
      <Tooltip
        ref={tooltipRef}
        anchorSelect="menu"
        openEvents={{}}
        closeEvents={{}}
        afterShow={tooltipShownHandler}
        position={tooltipPosition}
        place="right"
      >
        Select a sample to view a circuit.
      </Tooltip>
    </section>
  );
}

export { Intro };
