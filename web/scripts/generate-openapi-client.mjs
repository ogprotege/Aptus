import { readFile, mkdir, writeFile } from "node:fs/promises";
import { fileURLToPath } from "node:url";
import openapiTS, { astToString, COMMENT_HEADER } from "openapi-typescript";

const repositoryRoot = fileURLToPath(new URL("../..", import.meta.url));
const schemaUrl = new URL("../../docs/reference/openapi.v1.json", import.meta.url);
const outputUrl = new URL("../src/generated/openapi.ts", import.meta.url);
const check = process.argv.slice(2).includes("--check");
const unknownArguments = process.argv.slice(2).filter((argument) => argument !== "--check");

if (unknownArguments.length) {
  console.error(`Unknown OpenAPI client generator arguments: ${unknownArguments.join(" ")}`);
  process.exit(2);
}

const nodes = await openapiTS(schemaUrl, {
  alphabetize: true,
  silent: true,
});
const rendered = COMMENT_HEADER + astToString(nodes);

if (check) {
  let current;
  try {
    current = await readFile(outputUrl, "utf8");
  } catch (error) {
    if (error && typeof error === "object" && "code" in error && error.code === "ENOENT") {
      console.error("Missing generated OpenAPI client: web/src/generated/openapi.ts");
      process.exit(1);
    }
    throw error;
  }
  if (current !== rendered) {
    console.error(
      "Generated OpenAPI client is stale. Run `npm run openapi:generate` from "
      + `${repositoryRoot}/web.`,
    );
    process.exit(1);
  }
} else {
  await mkdir(new URL("../src/generated/", import.meta.url), { recursive: true });
  await writeFile(outputUrl, rendered, "utf8");
}
