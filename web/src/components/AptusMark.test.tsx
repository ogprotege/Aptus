import { render, screen } from "@testing-library/react";
import { describe, expect, it } from "vitest";
import { AptusMark } from "./AptusMark";

describe("AptusMark", () => {
  it("renders the selected Tile geometry and accessible name", () => {
    const { container } = render(<AptusMark className="brand-mark" />);

    expect(screen.getByRole("img", { name: "Aptus tile mark" })).toBeInTheDocument();
    const tile = container.querySelector(".aptus-mark-tile");
    expect(tile).toHaveAttribute("x", "72");
    expect(tile).toHaveAttribute("y", "72");
    expect(tile).toHaveAttribute("width", "880");
    expect(tile).toHaveAttribute("height", "880");
    expect(tile).toHaveAttribute("rx", "132");
  });
});
