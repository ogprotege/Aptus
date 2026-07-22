import type { BundleFile } from "../types";

interface ArtifactTreeProps {
  bundleDir: string;
  files: Array<string | BundleFile>;
}

export function ArtifactTree({ bundleDir, files }: ArtifactTreeProps) {
  const folderName = bundleDir.split("/").filter(Boolean).at(-1) ?? "aptus-bundle";
  return (
    <section className="artifact-tree" aria-labelledby="artifact-tree-title">
      <div className="section-heading-row compact-heading">
        <div>
          <p className="eyebrow">Versioned output</p>
          <h2 id="artifact-tree-title">Artifact tree</h2>
        </div>
        <span className="file-count">{files.length} files</span>
      </div>
      <div className="tree-root">
        <span className="folder-icon" aria-hidden="true">▾</span>
        <strong>{folderName}/</strong>
      </div>
      <ul className="tree-files">
        {files.map((file) => {
          const value = typeof file === "string" ? { path: file } : file;
          return (
            <li key={value.path}>
              <span className="file-branch" aria-hidden="true">├─</span>
              <span className="file-name">{value.path}</span>
              {value.size_bytes !== undefined ? (
                <small>{Math.max(1, Math.round(value.size_bytes / 1024))} KiB</small>
              ) : null}
            </li>
          );
        })}
      </ul>
    </section>
  );
}
