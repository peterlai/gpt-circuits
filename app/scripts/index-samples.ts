import fs from 'fs';
import { join } from 'path';

type SampleEntry  = {
    name: string;
    text: string;
    decodedTokens: string;
    targetIdx: number;
}
const modelsDir = 'public/samples';
const index: {[key: string]: SampleEntry[]} = {};

for (const modelName of fs.readdirSync(modelsDir)) {
    // Use whitelist.
    const approvedModels = ["toy", "toy-resample", "toy-zero", "baby"];
    if (!approvedModels.includes(modelName)) continue;

    // e.g. toy/
    const modelPath = join(modelsDir, modelName);
    if (!fs.statSync(modelPath).isDirectory()) continue;

    index[modelName] = [];

    const samplesPath = join(modelPath, "samples");
    if (!fs.existsSync(samplesPath) || !fs.statSync(samplesPath).isDirectory()) continue;

    // e.g. toy/samples/123456/
    for (const sampleName of fs.readdirSync(samplesPath)) {
        const samplePath = join(samplesPath, sampleName);
        if (!fs.statSync(samplePath).isDirectory()) continue;

        const json = await fs.promises.readFile(join(samplePath, "data.json"), "utf-8");
        const sampleData = JSON.parse(json);

        let decodedTokens = sampleData.decoded_tokens;
        if (decodedTokens === undefined) {
            // Backwards-compatible with existing data - can remove when all regenerated
            decodedTokens = [...sampleData.text];
        }

        index[modelName].push({
            name: sampleName,
            text: sampleData.text,
            decodedTokens: decodedTokens,
            targetIdx: sampleData.target_idx,
        });
    }
}

const outputPath = join(modelsDir, "index.json");
const json = JSON.stringify(index);
fs.writeFileSync(outputPath, json, "utf-8");
