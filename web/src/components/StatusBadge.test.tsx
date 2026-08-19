import { render, screen } from "@testing-library/react";
import { describe, expect, it } from "vitest";
import { StatusBadge } from "./StatusBadge";

describe("StatusBadge", () => {
  it("maps omitted states to the omitted tone", () => {
    const { container } = render(<StatusBadge state="omitted" />);
    expect(container.firstChild).toHaveClass("status-omitted");
    expect(screen.getByText("omitted")).toBeInTheDocument();
  });

  it("keeps unsupported on the negative tone", () => {
    const { container } = render(<StatusBadge state="unsupported" />);
    expect(container.firstChild).toHaveClass("status-negative");
  });

  it("keeps conditional on the warning tone", () => {
    const { container } = render(<StatusBadge state="conditional" />);
    expect(container.firstChild).toHaveClass("status-warning");
  });
});
