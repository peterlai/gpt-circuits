import { useAtom } from "jotai";
import { useEffect, useRef } from "react";
import { Tooltip, TooltipRefProps } from "react-tooltip";
import { isMenuOpenAtom, isMobile } from "../../stores/Navigation";
import { getEmbeddedSamplePath } from "../../views/App/urls";
import { SampleAnchor } from "../Home/Sample";
import "./Intro.scss";
function Intro() {
  const modelId = "toy-v0";
  const sampleId = "val.0.69248.76";
  const version = "0.1";
  const selectionKey = "0.2";
  const embeddedSamplePath = `#${getEmbeddedSamplePath(modelId, sampleId, version, selectionKey)}`;
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
        <span>GPT-2 Circuits</span>
      </h1>
      <p>
        This{" "}
        <a href="#/" onClick={openAppHandler}>
          app
        </a>{" "}
        visualizes “circuits” that have been extracted from an LLM using the GPT-2 architecture. The
        nodes in these circuits represent groups of features that collectively respond to clear
        patterns in the text. Here’s an example node that picks out five-to-six letter words ending
        in “s”.
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
        <iframe title="Embedded Circuit" src={embeddedSamplePath} height="420px" />
      </figure>
      <p>
        This research builds upon a{" "}
        <a href="https://peterlai.github.io/gpt-mri/" target="_blank" rel="noreferrer">
          previous version
        </a>{" "}
        of this project in which we extract circuits from 4 &amp; 6 layer character-based GPT-2
        models.
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
