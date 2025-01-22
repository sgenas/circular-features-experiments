// PCACatterPlot.tsx
import React, { useEffect, useRef, useState } from "react";
import * as d3 from "d3";
import { LayerData, PCARawData, Point } from "./types";

interface PCAScatterPlotProps {
  data: PCARawData;
  width?: number;
  height?: number;
}

const PCAScatterPlot: React.FC<PCAScatterPlotProps> = ({
  data,
  width = 800,
  height = 600,
}) => {
  const svgRef = useRef<SVGSVGElement>(null);
  const [currentLayer, setCurrentLayer] = useState<number>(
    Math.min(...Object.keys(data).map(Number))
  );

  const margin = { top: 60, right: 50, bottom: 50, left: 50 };
  const innerWidth = width - margin.left - margin.right;
  const innerHeight = height - margin.top - margin.bottom;

  const getBaseNote = (label: string): string =>
    label.split(" ").pop()!.charAt(0);
  const isBaseNote = (label: string): boolean => label.length === 1;

  useEffect(() => {
    if (!svgRef.current) return;

    const svg = d3.select(svgRef.current);
    svg.selectAll("*").remove(); // Clear previous render

    const currentData = data[currentLayer];

    // Create scales
    const xScale = d3
      .scaleLinear()
      .domain(
        d3.extent(currentData.states_pca, (d) => d[0]) as [number, number]
      )
      .range([margin.left, width - margin.right])
      .nice();

    const yScale = d3
      .scaleLinear()
      .domain(
        d3.extent(currentData.states_pca, (d) => d[1]) as [number, number]
      )
      .range([height - margin.bottom, margin.top])
      .nice();

    const colorScale = d3
      .scaleOrdinal<string>()
      .domain(["C", "D", "E", "F", "G", "A", "B"])
      .range(d3.schemeCategory10);

    // Create main group
    const g = svg.append("g");

    // Set up zoom
    const zoom = d3
      .zoom<SVGSVGElement, unknown>()
      .scaleExtent([0.5, 5])
      .on("zoom", (event: d3.D3ZoomEvent<SVGSVGElement, unknown>) => {
        const transform = event.transform;

        // Transform points and labels
        g.selectAll(".points-group").attr("transform", transform.toString());
        g.selectAll(".zero-lines").attr("transform", transform.toString());

        // Update axes
        const xScaleT = transform.rescaleX(xScale);
        const yScaleT = transform.rescaleY(yScale);

        xAxisGroup.call(xAxis.scale(xScaleT));
        yAxisGroup.call(yAxis.scale(yScaleT));

        // Update zero lines
        zeroLines
          .select("line.zero-line:nth-child(1)")
          .attr("y1", yScaleT(0))
          .attr("y2", yScaleT(0));

        zeroLines
          .select("line.zero-line:nth-child(2)")
          .attr("x1", xScaleT(0))
          .attr("x2", xScaleT(0));
      });

    svg.call(zoom);

    // Create axes
    const xAxis = d3.axisBottom(xScale);
    const yAxis = d3.axisLeft(yScale);

    const xAxisGroup = g
      .append("g")
      .attr("class", "x-axis")
      .attr("transform", `translate(0,${height - margin.bottom})`)
      .call(xAxis);

    const yAxisGroup = g
      .append("g")
      .attr("class", "y-axis")
      .attr("transform", `translate(${margin.left},0)`)
      .call(yAxis);

    // Add zero lines
    const zeroLines = g.append("g").attr("class", "zero-lines");

    zeroLines
      .append("line")
      .attr("class", "zero-line")
      .attr("x1", margin.left)
      .attr("x2", width - margin.right)
      .attr("y1", yScale(0))
      .attr("y2", yScale(0));

    zeroLines
      .append("line")
      .attr("class", "zero-line")
      .attr("x1", xScale(0))
      .attr("x2", xScale(0))
      .attr("y1", margin.top)
      .attr("y2", height - margin.bottom);

    // Create tooltip
    const tooltip = d3
      .select("body")
      .append("div")
      .attr("class", "tooltip")
      .style("opacity", 0);

    // Add points and labels
    const pointsGroup = g.append("g").attr("class", "points-group");

    pointsGroup
      .selectAll("circle")
      .data(currentData.states_pca)
      .enter()
      .append("circle")
      .attr("class", "point")
      .attr("cx", (d) => xScale(d[0]))
      .attr("cy", (d) => yScale(d[1]))
      .attr("r", (_, i) => (isBaseNote(currentData.display_labels[i]) ? 10 : 8))
      .attr("fill", (_, i) =>
        colorScale(getBaseNote(currentData.display_labels[i]))
      )
      .on("mouseover", (event, d) => {
        const index = currentData.states_pca.indexOf(d);
        const label = currentData.display_labels[index];
        d3.select(event.currentTarget)
          .transition()
          .duration(200)
          .attr("r", isBaseNote(label) ? 12 : 10);
        tooltip.transition().duration(200).style("opacity", 0.9);
        tooltip
          .html(label)
          .style("left", `${event.pageX + 10}px`)
          .style("top", `${event.pageY - 10}px`);
      })
      .on("mouseout", (event) => {
        const index = currentData.states_pca.indexOf(
          d3.select(event.currentTarget).datum() as Point
        );
        const label = currentData.display_labels[index];
        d3.select(event.currentTarget)
          .transition()
          .duration(200)
          .attr("r", isBaseNote(label) ? 10 : 8);
        tooltip.transition().duration(500).style("opacity", 0);
      });

    pointsGroup
      .selectAll(".label")
      .data(currentData.states_pca)
      .enter()
      .append("text")
      .attr("class", "label")
      .attr("x", (d) => xScale(d[0]))
      .attr("y", (d) => yScale(d[1]) - 12)
      .attr("text-anchor", "middle")
      .text((_, i) => currentData.display_labels[i]);

    // Add title and axis labels
    svg
      .append("text")
      .attr("class", "plot-title")
      .attr("x", width / 2)
      .attr("y", margin.top / 2)
      .attr("text-anchor", "middle")
      .text(currentData.model_config);

    svg
      .append("text")
      .attr("class", "axis-label")
      .attr("x", width / 2)
      .attr("y", height - margin.bottom / 3)
      .attr("text-anchor", "middle")
      .text("PCA Component 1");

    svg
      .append("text")
      .attr("class", "axis-label")
      .attr("transform", "rotate(-90)")
      .attr("x", -(height / 2))
      .attr("y", margin.left / 3)
      .attr("text-anchor", "middle")
      .text("PCA Component 2");

    // Add legend
    const legend = svg
      .append("g")
      .attr("text-anchor", "start")
      .selectAll("g")
      .data(["C", "D", "E", "F", "G", "A", "B"])
      .enter()
      .append("g")
      .attr(
        "transform",
        (_, i) => `translate(${width - margin.right},${margin.top + i * 20})`
      );

    legend
      .append("circle")
      .attr("cx", 0)
      .attr("cy", 0)
      .attr("r", 5)
      .attr("fill", colorScale);

    legend
      .append("text")
      .attr("x", 10)
      .attr("y", 3)
      .text((d) => d);

    return () => {
      tooltip.remove();
    };
  }, [currentLayer, data, width, height, margin]);

  return (
    <div className="pca-scatter-plot">
      <div className="slider-container">
        <div className="layer-display">Layer: {currentLayer}</div>
        <input
          type="range"
          min={Math.min(...Object.keys(data).map(Number))}
          max={Math.max(...Object.keys(data).map(Number))}
          step={5}
          value={currentLayer}
          onChange={(e) => setCurrentLayer(Number(e.target.value))}
          className="w-[300px]"
        />
      </div>
      <svg ref={svgRef} width={width} height={height} className="bg-white" />
      <style jsx>{`
        .tooltip {
          position: absolute;
          padding: 8px;
          background: white;
          border: 1px solid #ddd;
          border-radius: 4px;
          pointer-events: none;
          font-family: "Inter", sans-serif;
          font-size: 12px;
        }
        .point {
          fill-opacity: 0.8;
          stroke: #fff;
          stroke-width: 1px;
        }
        .point:hover {
          fill-opacity: 1;
        }
        .label {
          font-family: "Inter", sans-serif;
          font-size: 10px;
          pointer-events: none;
        }
        .zero-line {
          stroke: #000;
          stroke-dasharray: 4;
          stroke-width: 1px;
          opacity: 0.3;
        }
        .axis-label {
          font-family: "Inter", sans-serif;
          font-size: 14px;
        }
        .plot-title {
          font-family: "Inter", sans-serif;
          font-size: 20px;
          font-weight: 600;
        }
        .slider-container {
          display: flex;
          flex-direction: column;
          align-items: center;
          margin: 20px 0;
          gap: 10px;
          font-family: "Inter", sans-serif;
        }
        .layer-display {
          font-size: 16px;
          font-weight: 500;
        }
      `}</style>
    </div>
  );
};

export default PCAScatterPlot;

// Usage example:
//
// import { PCARawData } from './types';
// import PCAScatterPlot from './PCAScatterPlot';
//
// const data: PCARawData = {
//   5: {
//     experiment_name: "musical_note_flat_sharp",
//     model_config: "Gemma-2-2B",
//     layer: 5,
//     display_labels: [...],
//     states_pca: [...],
//     explained_variance: [...]
//   },
//   // ... other layers
// };
//
// const App: React.FC = () => (
//   <PCAScatterPlot data={data} />
// );
