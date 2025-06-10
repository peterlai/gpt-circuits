import classNames from "classnames";
import { useAtom, useAtomValue, useSetAtom } from "jotai";
import { BiArrowFromRight } from "react-icons/bi";
import { FaLayerGroup } from "react-icons/fa6";

import { modelIdAtom, sampleIdAtom, versionAtom } from "../stores/Graph";
import { isMenuOpenAtom, modelOptionsAtom, SampleOption } from "../stores/Navigation";
import { getAboutPath, getInspectSamplePath } from "../views/App/urls";

import "./Menu.scss";

function Menu() {
  const setIsMenuOpen = useSetAtom(isMenuOpenAtom);
  const [{ data: modelOptions }] = useAtom(modelOptionsAtom);

  const shakespeareSampleIdToOptions = new Map<string, SampleOption[]>();
  const tinyStoriesSampleIdToOptions = new Map<string, SampleOption[]>();

  // For each model option, group sample options by sample ID and dataset type.
  Object.values(modelOptions ?? {}).forEach((modelOption) => {
    Object.entries(modelOption.sampleOptions).forEach(([sampleId, sampleOption]) => {
      // Determine dataset type based on tokenization strategy
      // Shakespeare uses character-level tokenization (all tokens are single characters)
      // TinyStories uses subword tokenization (has multi-character tokens)
      const tokens = sampleOption.decodedTokens || [];
      const isShakespeare = !tokens.some((token) => token.length > 1);
      const targetMap = isShakespeare ? shakespeareSampleIdToOptions : tinyStoriesSampleIdToOptions;

      const options = targetMap.get(sampleId) ?? [];
      options.push(sampleOption);
      targetMap.set(sampleId, options);
    });
  });

  // Sort sample IDs numerically if possible.
  const sortSampleIds = (sampleIds: string[]) => {
    return sampleIds.sort((a, b) => {
      // Check if ID is in the format "x.0.0.0"
      const aParts = a.split(".");
      const bParts = b.split(".");
      if (aParts.length === 4 && bParts.length === 4) {
        // Compare the first part as string
        const splitComparison = aParts[0].localeCompare(bParts[0]);
        if (splitComparison !== 0) return splitComparison;

        // Compare the remaining parts as numbers
        for (let i = 1; i < 4; i++) {
          const aNum = parseInt(aParts[i]);
          const bNum = parseInt(bParts[i]);
          if (aNum !== bNum) return aNum - bNum;
        }

        // If all parts are equal, return 0.
        return 0;
      } else {
        // Otherwise, sort as strings.
        return a.localeCompare(b);
      }
    });
  };

  const sortedShakespeareSampleIds = sortSampleIds(Array.from(shakespeareSampleIdToOptions.keys()));
  const sortedTinyStoriesSampleIds = sortSampleIds(Array.from(tinyStoriesSampleIdToOptions.keys()));

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
          {[
            {
              name: "TinyStories",
              url: "https://huggingface.co/datasets/roneneldan/TinyStories",
              sampleIds: sortedTinyStoriesSampleIds,
              sampleMap: tinyStoriesSampleIdToOptions,
            },
            {
              name: "Shakespeare",
              url: "https://huggingface.co/datasets/karpathy/tiny_shakespeare",
              sampleIds: sortedShakespeareSampleIds,
              sampleMap: shakespeareSampleIdToOptions,
            },
          ].map(
            ({ name, url, sampleIds, sampleMap }) =>
              sampleIds.length > 0 && (
                <div key={name}>
                  <h3>
                    <a href={url} target="about:blank">
                      {name}
                    </a>{" "}
                    Samples
                  </h3>
                  <ul className="samples">
                    {sampleIds.map((sampleId) => (
                      <SampleMenuItem key={sampleId} sampleOptions={sampleMap.get(sampleId)!} />
                    ))}
                  </ul>
                </div>
              )
          )}
        </section>
      </div>
    </menu>
  );
}

function SampleMenuItem({ sampleOptions }: { sampleOptions: SampleOption[] }) {
  const selectedModelId = useAtomValue(modelIdAtom);
  const selectedSampleId = useAtomValue(sampleIdAtom);
  const selectedVersion = useAtomValue(versionAtom);

  // Does one of the sample options reference the selected model?
  const similarSampleOption = sampleOptions.find((o) => o.modelId === selectedModelId);

  // Is the selected model recognized?
  const isCustomModel = sampleOptions.some((o) => o.modelId === selectedModelId);

  // Sort sample options by layer count.
  const sortedSampleOptions = sampleOptions.sort((a, b) => a.layerCount - b.layerCount);
  const lastSampleOption = sortedSampleOptions[sortedSampleOptions.length - 1];
  const sampleId = lastSampleOption.id;

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
  if (isCustomModel) {
    // If model is unrecognized, infer sample URL from current states.
    defaultSampleUrl = getInspectSamplePath(selectedModelId, sampleId, selectedVersion);
  } else if (similarSampleOption) {
    // If a sample option exist for the selected model, use option. Try to match the selected version.
    const sampleVersion =
      similarSampleOption.versions.find((v) => v === selectedVersion) ??
      similarSampleOption.defaultVersion;
    defaultSampleUrl = getInspectSamplePath(selectedModelId, sampleId, sampleVersion);
  } else {
    // Otherwise, use the last sample option.
    defaultSampleUrl = getInspectSamplePath(
      lastSampleOption.modelId,
      sampleId,
      lastSampleOption.defaultVersion
    );
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
