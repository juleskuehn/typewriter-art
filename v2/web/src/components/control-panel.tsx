"use client";

import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select";
import { Slider } from "@/components/ui/slider";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import {
  Collapsible,
  CollapsibleContent,
  CollapsibleTrigger,
} from "@/components/ui/collapsible";
import { useOptimizerStore, useIsOptimizing } from "@/store/optimizer";
import { CHARSETS, LAYER_CONFIGS } from "@/lib/types";
import { Settings2, ChevronDown } from "lucide-react";
import { useState } from "react";

export function ControlPanel() {
  const config = useOptimizerStore((s) => s.config);
  const setConfig = useOptimizerStore((s) => s.setConfig);
  const optimize = useOptimizerStore((s) => s.optimize);
  const isOptimizing = useIsOptimizing();
  const [advancedOpen, setAdvancedOpen] = useState(false);

  return (
    <Card className="w-full max-w-sm">
      <CardHeader>
        <CardTitle className="text-lg flex items-center gap-2">
          <Settings2 className="w-4 h-4" />
          Settings
        </CardTitle>
      </CardHeader>
      <CardContent className="space-y-5">
        {/* Charset */}
        <div className="space-y-2">
          <label className="text-sm font-medium">Charset</label>
          <Select
            value={config.charset}
            onValueChange={(v) => v && setConfig("charset", v)}
            disabled={isOptimizing}
          >
            <SelectTrigger>
              <SelectValue />
            </SelectTrigger>
            <SelectContent>
              {CHARSETS.map((c) => (
                <SelectItem key={c.value} value={c.value}>
                  {c.label}
                </SelectItem>
              ))}
            </SelectContent>
          </Select>
        </div>

        {/* Layers */}
        <div className="space-y-2">
          <label className="text-sm font-medium">Layers</label>
          <Select
            value={config.layers}
            onValueChange={(v) => v && setConfig("layers", v)}
            disabled={isOptimizing}
          >
            <SelectTrigger>
              <SelectValue />
            </SelectTrigger>
            <SelectContent>
              {LAYER_CONFIGS.map((l) => (
                <SelectItem key={l.value} value={l.value}>
                  {l.label}
                </SelectItem>
              ))}
            </SelectContent>
          </Select>
        </div>

        {/* Search Algorithm */}
        <div className="space-y-2">
          <label className="text-sm font-medium">Algorithm</label>
          <Select
            value={config.search}
            onValueChange={(v) => v && setConfig("search", v as "simAnneal" | "greedy")}
            disabled={isOptimizing}
          >
            <SelectTrigger>
              <SelectValue />
            </SelectTrigger>
            <SelectContent>
              <SelectItem value="simAnneal">Simulated Annealing</SelectItem>
              <SelectItem value="greedy">Greedy</SelectItem>
            </SelectContent>
          </Select>
        </div>

        {/* Row Length */}
        <div className="space-y-2">
          <div className="flex justify-between">
            <label className="text-sm font-medium">Row Length</label>
            <span className="text-sm text-gray-500">{config.row_length} chars</span>
          </div>
          <Slider
            value={[config.row_length]}
            onValueChange={(v) => setConfig("row_length", Array.isArray(v) ? v[0] : (v as unknown as number))}
            min={10}
            max={80}
            step={1}
            disabled={isOptimizing}
          />
        </div>

        {/* Num Loops */}
        <div className="space-y-2">
          <div className="flex justify-between">
            <label className="text-sm font-medium">Loops</label>
            <span className="text-sm text-gray-500">{config.num_loops}</span>
          </div>
          <Slider
            value={[config.num_loops]}
            onValueChange={(v) => setConfig("num_loops", Array.isArray(v) ? v[0] : (v as unknown as number))}
            min={1}
            max={30}
            step={1}
            disabled={isOptimizing}
          />
        </div>

        {/* Asymmetry */}
        <div className="space-y-2">
          <div className="flex justify-between">
            <label className="text-sm font-medium">Asymmetry</label>
            <span className="text-sm text-gray-500">{config.asymmetry.toFixed(2)}</span>
          </div>
          <Slider
            value={[config.asymmetry]}
            onValueChange={(v) => setConfig("asymmetry", Array.isArray(v) ? v[0] : (v as unknown as number))}
            min={0}
            max={1}
            step={0.01}
            disabled={isOptimizing}
          />
        </div>

        {/* Advanced Options */}
        <Collapsible open={advancedOpen} onOpenChange={setAdvancedOpen}>
          <CollapsibleTrigger className="flex items-center gap-1 text-sm text-gray-500 hover:text-gray-700 transition-colors w-full">
            <ChevronDown
              className={`w-4 h-4 transition-transform ${advancedOpen ? "rotate-180" : ""}`}
            />
            Advanced Options
          </CollapsibleTrigger>
          <CollapsibleContent className="space-y-4 pt-3">
            {/* Init Mode */}
            <div className="space-y-2">
              <label className="text-sm font-medium">Init Mode</label>
              <Select
                value={config.init_mode}
                onValueChange={(v) => v && setConfig("init_mode", v as "random" | "blank")}
                disabled={isOptimizing}
              >
                <SelectTrigger>
                  <SelectValue />
                </SelectTrigger>
                <SelectContent>
                  <SelectItem value="random">Random</SelectItem>
                  <SelectItem value="blank">Blank</SelectItem>
                </SelectContent>
              </Select>
            </div>

            {/* Temperature */}
            <div className="space-y-2">
              <div className="flex justify-between">
                <label className="text-sm font-medium">Temperature</label>
                <span className="text-sm text-gray-500">
                  {config.init_temp.toFixed(4)}
                </span>
              </div>
              <Slider
                value={[config.init_temp]}
                onValueChange={(v) => setConfig("init_temp", Array.isArray(v) ? v[0] : (v as unknown as number))}
                min={0.0001}
                max={0.05}
                step={0.0001}
                disabled={isOptimizing}
              />
            </div>

            {/* Shuffle */}
            <label className="flex items-center gap-2 text-sm">
              <input
                type="checkbox"
                checked={config.shuffle}
                onChange={(e) => setConfig("shuffle", e.target.checked)}
                disabled={isOptimizing}
                className="rounded"
              />
              <span className="font-medium">Shuffle layer order</span>
            </label>
          </CollapsibleContent>
        </Collapsible>

        {/* Optimize Button */}
        <Button
          className="w-full"
          size="lg"
          onClick={optimize}
          disabled={isOptimizing}
        >
          {isOptimizing ? (
            <span className="flex items-center gap-2">
              <Spinner />
              Optimizing...
            </span>
          ) : (
            "Optimize"
          )}
        </Button>
      </CardContent>
    </Card>
  );
}

function Spinner() {
  return (
    <svg
      className="animate-spin h-4 w-4"
      xmlns="http://www.w3.org/2000/svg"
      fill="none"
      viewBox="0 0 24 24"
    >
      <circle
        className="opacity-25"
        cx="12" cy="12" r="10"
        stroke="currentColor"
        strokeWidth="4"
      />
      <path
        className="opacity-75"
        fill="currentColor"
        d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4z"
      />
    </svg>
  );
}
