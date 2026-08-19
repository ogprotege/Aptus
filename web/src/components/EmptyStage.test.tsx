import { render, screen } from "@testing-library/react";
import { describe, expect, it, vi } from "vitest";
import { EmptyStage } from "./EmptyStage";

describe("EmptyStage", () => {
  it("uses a secondary action when the empty state is omitted", () => {
    render(
      <EmptyStage title="No last call recorded." actionLabel="Stay on Run" onAction={() => undefined} tone="omitted">
        Missing last call is not Use.
      </EmptyStage>,
    );
    const section = screen.getByText("No last call recorded.").closest("section");
    expect(section).toHaveClass("evidence-omitted");
    expect(screen.getByRole("button", { name: "Stay on Run" })).toHaveClass("button-secondary");
    expect(screen.getByRole("button", { name: "Stay on Run" })).not.toHaveClass("button-primary");
  });

  it("keeps the default empty stage on the path button", () => {
    const onAction = vi.fn();
    render(
      <EmptyStage title="Need a plan" actionLabel="Back to Facts" onAction={onAction}>
        Compile waits on a plan.
      </EmptyStage>,
    );
    expect(screen.getByRole("button", { name: "Back to Facts" })).toHaveClass("button-primary");
  });
});
