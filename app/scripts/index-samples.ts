import fs from 'fs';
import { join } from 'path';

// Parse command line arguments
const args = process.argv.slice(2);
const buildMode = args.includes('build');

let modelsDir: string;
let indexedModelPrefixes: string[]; // Show these in the menu
let hiddenModelPrefixes: string[]; // Hide these in the UI

// Set directory and prefixes based on build mode
if (buildMode) {
    console.log("Building index for deployment");
    modelsDir = buildMode ? 'build/samples' : 'public/samples';
    indexedModelPrefixes = ["toy-v0"];
    hiddenModelPrefixes = ["comparisons"];
} else {
    modelsDir = 'public/samples';
    indexedModelPrefixes = ["toy"];
    hiddenModelPrefixes = ["toy-v0"];
}

// If building for deploying, clean up build directory
if (buildMode) {
    for (const modelName of fs.readdirSync(modelsDir)) {
        // Preserve everything that is indexed or hidden.
        if (indexedModelPrefixes.some(prefix => modelName.startsWith(prefix))) continue;
        if (hiddenModelPrefixes.some(prefix => modelName.startsWith(prefix))) continue;

        // Remove everything else.
        const modelPath = join(modelsDir, modelName);
        if (fs.statSync(modelPath).isDirectory()) {
            fs.rmSync(modelPath, { recursive: true, force: true });
        }
    }
}

type SampleEntry  = {
    name: string;
    versions: string[];
    text: string;
    decodedTokens: string;
    targetIdx: number;
}
const index: {[key: string]: SampleEntry[]} = {};

for (const modelName of fs.readdirSync(modelsDir)) {
    // Use allowlist.
    if (!indexedModelPrefixes.some(prefix => modelName.startsWith(prefix))) continue;

    // Use blocklist.
    if (hiddenModelPrefixes.some(prefix => modelName.startsWith(prefix))) continue;

    // e.g. toy/
    const modelPath = join(modelsDir, modelName);
    if (!fs.statSync(modelPath).isDirectory()) continue;

    index[modelName] = [];

    // e.g. toy/samples
    const samplesPath = join(modelPath, "samples");
    if (!fs.existsSync(samplesPath) || !fs.statSync(samplesPath).isDirectory()) continue;

    // e.g. train.0.0.51
    for (const sampleDirname of fs.readdirSync(samplesPath)) {
        // e.g. toy/samples/train.0.0.51
        const samplePath = join(samplesPath, sampleDirname);
        if (!fs.statSync(samplePath).isDirectory()) continue;

        // Store version names
        const versions: string[] = [];
        for (const versionDirname of fs.readdirSync(samplePath)) {
            // Ignore files that start with a dot
            if (versionDirname.startsWith('.')) continue;
            versions.push(versionDirname);
        }

        // Fetch data from first version
        if (versions.length === 0) continue;
        const firstVersion = versions[0];

        // e.g. toy/samples/train.0.0.51/0.2
        const versionPath = join(samplePath, firstVersion);
        if (!fs.statSync(versionPath).isDirectory()) continue;

        // e.g. toy/samples/train.0.0.51/0.2/data.json
        const dataPath = join(versionPath, "data.json");
        if (!fs.existsSync(dataPath)) continue;

        // Read data
        const data = JSON.parse(await fs.promises.readFile(dataPath, "utf-8"));

        index[modelName].push({
            name: sampleDirname,
            versions: versions,
            text: data.text,
            decodedTokens: data.decodedTokens,
            targetIdx: data.targetIdx,
        });
    }
}

const outputPath = join(modelsDir, "index.json");
const json = JSON.stringify(index);
fs.writeFileSync(outputPath, json, "utf-8");
