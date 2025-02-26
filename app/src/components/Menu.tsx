import classNames from "classnames";
import { useAtom, useAtomValue, useSetAtom } from "jotai";
import { BiArrowFromRight } from "react-icons/bi";
import { FaLayerGroup } from "react-icons/fa6";

import { modelIdAtom, sampleIdAtom } from "../stores/Graph";
import { isMenuOpenAtom, modelOptionsAtom, SampleOption } from "../stores/Navigation";
import { getAboutPath, getInspectSamplePath } from "../views/App/urls";

import "./Menu.scss";

function Menu() {
  const setIsMenuOpen = useSetAtom(isMenuOpenAtom);
  const [{ data: modelOptions }] = useAtom(modelOptionsAtom);

  const shakespeareSampleIdToOptions = new Map<string, SampleOption[]>();

  // For each model option, group sample options by sample ID.
  Object.values(modelOptions ?? {}).forEach((modelOption) => {
    Object.entries(modelOption.sampleOptions).forEach(([sampleId, sampleOption]) => {
      const options = shakespeareSampleIdToOptions.get(sampleId) ?? [];
      options.push(sampleOption);
      shakespeareSampleIdToOptions.set(sampleId, options);
    });
  });

  // Sort sample IDs numerically if possible.
  const sortedShakespeareSampleIds = Array.from(shakespeareSampleIdToOptions.keys()).sort(
    (a, b) => {
      const aId = isNaN(parseInt(a)) ? 0 : parseInt(a);
      const bId = isNaN(parseInt(b)) ? 0 : parseInt(b);
      return aId - bId;
    }
  );

  return (
    <menu>
      <div className="inner">
        <header>
          <span>Menu</span>
          <BiArrowFromRight className="close-menu" onClick={() => setIsMenuOpen(false)} />
        </header>
        <section>
          <h3>What is this?</h3>
          <p>
            This app uses sparse autoencoders and feature ablation to map the inner circuitry of an
            LLM. <a href={`#${getAboutPath()}`}>Learn more &raquo;</a>
          </p>
          <h3>
            <a
              href="https://huggingface.co/datasets/karpathy/tiny_shakespeare"
              target="about:blank"
            >
              Shakespeare
            </a>{" "}
            Samples
          </h3>
          <ul className="samples">
            {sortedShakespeareSampleIds.map((sampleId) => (
              <SampleMenuItem
                key={sampleId}
                sampleOptions={shakespeareSampleIdToOptions.get(sampleId)!}
              />
            ))}
          </ul>
        </section>
      </div>
    </menu>
  );
}

function SampleMenuItem({ sampleOptions }: { sampleOptions: SampleOption[] }) {
  const selectedModelId = useAtomValue(modelIdAtom);
  const selectedSampleId = useAtomValue(sampleIdAtom);

  // Does one of the sample options reference the selected model?
  const hasSampleOption = sampleOptions.find((o) => o.modelId === selectedModelId);

  // Is the selected model a custom model?
  const isCustomModel = selectedModelId.indexOf("-") !== -1;

  // Sort sample options by layer count.
  const sortedSampleOptions = sampleOptions.sort((a, b) => a.layerCount - b.layerCount);
  const lastSampleOption = sortedSampleOptions[sortedSampleOptions.length - 1];
  const sampleId = lastSampleOption.id;
  const sampleVersion = lastSampleOption.defaultVersion;

  const targetIdx = lastSampleOption.targetIdx;
  const padLeft = 10;
  const padRight = 20;
  const decodedTokens = lastSampleOption.decodedTokens.map((t) => t.replaceAll("\n", "⏎"));
  const leftOfTarget = decodedTokens.slice(0, targetIdx).join("").slice(-padLeft).padStart(padLeft);
  const rightOfTarget = decodedTokens
    .slice(targetIdx + 1)
    .join("")
    .slice(0, padRight);
  const targetToken = decodedTokens[targetIdx];

  // Set default sample URL, which is navigated to when the menu item is clicked.
  let defaultSampleUrl: string;
  if (hasSampleOption || isCustomModel) {
    // If a sample option exist for the selected model (or if using custom model), use selected model.
    defaultSampleUrl = getInspectSamplePath(selectedModelId, sampleId, sampleVersion);
  } else {
    // Otherwise, use the last sample option.
    defaultSampleUrl = getInspectSamplePath(lastSampleOption.modelId, sampleId, sampleVersion);
  }

  return (
    <li
      className={classNames({
        selected: sampleId === selectedSampleId.split("-")[0],
      })}
      onClick={() => {
        window.location.hash = defaultSampleUrl;
      }}
    >
      <pre className="sample">
        <span>{leftOfTarget}</span>
        <em>{targetToken}</em>
        <span>{rightOfTarget}</span>
      </pre>
      <span className="links">
        {sortedSampleOptions.map((sampleOption, i) => (
          <SampleOptionLink key={i} sampleOption={sampleOption} />
        ))}
      </span>
    </li>
  );
}

function SampleOptionLink({ sampleOption }: { sampleOption: SampleOption }) {
  const selectedSampleId = useAtomValue(sampleIdAtom);
  const selectedModelId = useAtomValue(modelIdAtom);
  const isSelected =
    sampleOption.id === selectedSampleId && sampleOption.modelId === selectedModelId;

  return (
    <a
      className={classNames({
        selected: isSelected,
      })}
      href={`#${getInspectSamplePath(
        sampleOption.modelId,
        sampleOption.id,
        sampleOption.defaultVersion
      )}`}
    >
      <FaLayerGroup className="icon" />
      <span>{sampleOption.layerCount}</span>
    </a>
  );
}

export { Menu };
