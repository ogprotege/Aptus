import { render, screen } from "@testing-library/react";
import { describe, expect, it } from "vitest";
import { AptusMark } from "./AptusMark";

describe("AptusMark", () => {
  it("renders the refined calibrated A without a baked tile", () => {
    const { container } = render(<AptusMark className="brand-mark" />);

    expect(screen.getByRole("img", { name: "Aptus calibrated A" })).toBeInTheDocument();
    expect(container.querySelector("rect")).not.toBeInTheDocument();

    const paths = container.querySelectorAll("path");
    expect(paths).toHaveLength(2);
    expect(paths[0]).toHaveAttribute(
      "d",
      "M226 806 L460 244 C470 217 489 202 512 202 C535 202 554 217 564 244 L798 806",
    );
    expect(paths[0]).toHaveAttribute("stroke-width", "88");
    expect(paths[0]).toHaveAttribute("stroke-linecap", "round");
    expect(paths[1]).toHaveAttribute("d", "M256 608 H768");
    expect(paths[1]).toHaveAttribute("stroke-width", "64");
  });
});
