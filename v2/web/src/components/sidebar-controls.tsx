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
import {
  Collapsible,
  CollapsibleContent,
  CollapsibleTrigger,
} from "@/components/ui/collapsible";
import { useOptimizerStore, useActiveItem } from "@/store/optimizer";
import { CHARSETS, LAYER_CONFIGS } from "@/lib/types";
import { Settings2, ChevronDown, Loader2 } from "lucide-react";
import { useState } from "react";

export function SidebarControls() {
  const activeItem = useActiveItem();
  const setConfig = useOptimizerStore((s) => s.setConfig);
  const optimize = useOptimizerStore((s) => s.optimize);
  const progressStatus = useOptimizerStore((s) => s.progressStatus);
  const [busy, setBusy] = useState(false);
  const [advancedOpen, setAdvancedOpen] = useState(false);
  const [error, setError] = useState<string | null>(null);

  if (!activeItem) {
    return (
      <div className="w-72 p-6 border-l border-gray-200 bg-gray-50 flex items-center justify-center">
        <p className="text-sm text-gray-400 text-center">
          Drop or paste an image to begin
        </p>
      </div>
    );
  }

  const { config, result } = activeItem;

  const handleOptimize = async () => {
    setBusy(true);
    setError(null);
    try {
      await optimize();
    } catch (err) {
      // Surfaced next to the button — previously this vanished into the console.
      setError(err instanceof Error ? err.message : String(err));
    } finally {
      setBusy(false);
    }
  };

  return (
    <div className="w-72 flex flex-col border-l border-gray-200 bg-gray-50 overflow-y-auto">
      <div className="p-4 border-b border-gray-200">
        <h2 className="text-sm font-semibold text-gray-700 flex items-center gap-1.5">
          <Settings2 className="w-3.5 h-3.5" />
          Settings
        </h2>
      </div>

      <div className="flex-1 p-4 space-y-4 overflow-y-auto">
        {/* Charset */}
        <div className="space-y-1.5">
          <label className="text-xs font-medium text-gray-500">Charset</label>
          <Select
            value={config.charset}
            onValueChange={(v) => v && setConfig("charset", v)}
            disabled={busy}
          >
            <SelectTrigger className="h-8 text-xs">
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
        <div className="space-y-1.5">
          <label className="text-xs font-medium text-gray-500">Layers</label>
          <Select
            value={config.layers}
            onValueChange={(v) => v && setConfig("layers", v)}
            disabled={busy}
          >
            <SelectTrigger className="h-8 text-xs">
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

        {/* Algorithm */}
        <div className="space-y-1.5">
          <label className="text-xs font-medium text-gray-500">Algorithm</label>
          <Select
            value={config.search}
            onValueChange={(v) => v && setConfig("search", v as "simAnneal" | "greedy")}
            disabled={busy}
          >
            <SelectTrigger className="h-8 text-xs">
              <SelectValue />
            </SelectTrigger>
            <SelectContent>
              <SelectItem value="simAnneal">Simulated Annealing</SelectItem>
              <SelectItem value="greedy">Greedy</SelectItem>
            </SelectContent>
          </Select>
        </div>

        {/* Row Length */}
        <div className="space-y-1.5">
          <div className="flex justify-between">
            <label className="text-xs font-medium text-gray-500">Row length</label>
            <span className="text-xs text-gray-400">{config.row_length}</span>
          </div>
          <Slider
            value={[config.row_length]}
            onValueChange={(v) => setConfig("row_length", Array.isArray(v) ? v[0] : (v as unknown as number))}
            min={10} max={80} step={1}
            disabled={busy}
          />
        </div>

        {/* Loops */}
        <div className="space-y-1.5">
          <div className="flex justify-between">
            <label className="text-xs font-medium text-gray-500">Loops</label>
            <span className="text-xs text-gray-400">{config.num_loops}</span>
          </div>
          <Slider
            value={[config.num_loops]}
            onValueChange={(v) => setConfig("num_loops", Array.isArray(v) ? v[0] : (v as unknown as number))}
            min={1} max={30} step={1}
            disabled={busy}
          />
        </div>

        {/* Asymmetry */}
        <div className="space-y-1.5">
          <div className="flex justify-between">
            <label className="text-xs font-medium text-gray-500">Asymmetry</label>
            <span className="text-xs text-gray-400">{config.asymmetry.toFixed(2)}</span>
          </div>
          <Slider
            value={[config.asymmetry]}
            onValueChange={(v) => setConfig("asymmetry", Array.isArray(v) ? v[0] : (v as unknown as number))}
            min={0} max={1} step={0.01}
            disabled={busy}
          />
        </div>

        {/* Auto-align */}
        <label className="flex items-center gap-2 text-xs text-gray-600">
          <input
            type="checkbox"
            checked={config.align_enabled}
            onChange={(e) => setConfig("align_enabled", e.target.checked)}
            disabled={busy}
            className="rounded"
          />
          Auto-align to character grid
        </label>

        {/* Advanced */}
        <Collapsible open={advancedOpen} onOpenChange={setAdvancedOpen}>
          <CollapsibleTrigger className="flex items-center gap-1 text-xs text-gray-400 hover:text-gray-600 w-full">
            <ChevronDown className={`w-3 h-3 transition-transform ${advancedOpen ? "rotate-180" : ""}`} />
            Advanced
          </CollapsibleTrigger>
          <CollapsibleContent className="space-y-3 pt-2">
            <div className="space-y-1.5">
              <label className="text-xs font-medium text-gray-500">Init mode</label>
              <Select
                value={config.init_mode}
                onValueChange={(v) => v && setConfig("init_mode", v as "random" | "blank")}
                disabled={busy}
              >
                <SelectTrigger className="h-8 text-xs">
                  <SelectValue />
                </SelectTrigger>
                <SelectContent>
                  <SelectItem value="random">Random</SelectItem>
                  <SelectItem value="blank">Blank</SelectItem>
                </SelectContent>
              </Select>
            </div>

            <div className="space-y-1.5">
              <div className="flex justify-between">
                <label className="text-xs font-medium text-gray-500">Temperature</label>
                <span className="text-xs text-gray-400">{config.init_temp.toFixed(4)}</span>
              </div>
              <Slider
                value={[config.init_temp]}
                onValueChange={(v) => setConfig("init_temp", Array.isArray(v) ? v[0] : (v as unknown as number))}
                min={0.0001} max={0.05} step={0.0001}
                disabled={busy}
              />
            </div>

            <label className="flex items-center gap-2 text-xs text-gray-600">
              <input
                type="checkbox"
                checked={config.shuffle}
                onChange={(e) => setConfig("shuffle", e.target.checked)}
                disabled={busy}
                className="rounded"
              />
              Shuffle layer order
            </label>
          </CollapsibleContent>
        </Collapsible>
      </div>

      {/* Stats & buttons */}
      <div className="p-4 border-t border-gray-200 space-y-3">
        {result && (
          <div className="text-xs text-gray-400 space-y-0.5">
            <div>Error: {result.stats.final_error.toFixed(5)}</div>
            <div>Time: {(result.stats.total_time_ns / 1e6).toFixed(0)} ms</div>
            {result.alignment && (
              <div>
                Aligned: zoom {result.alignment.best.zoom}/{result.alignment.steps},
                shift ({result.alignment.best.shift_x}/{result.alignment.steps}, {result.alignment.best.shift_y}/{result.alignment.steps}) char
              </div>
            )}
          </div>
        )}

        {activeItem.sourceImageLost && (
          <p className="text-xs text-amber-600">
            The full-size image wasn&apos;t kept after the last page reload, so this
            entry can&apos;t be re-optimized. Re-upload the image to run it again.
          </p>
        )}

        {error && (
          <p className="text-xs text-red-600" role="alert">
            {error}
          </p>
        )}

        <Button
          className="w-full"
          size="sm"
          onClick={handleOptimize}
          disabled={busy || activeItem.sourceImageLost}
        >
          {busy ? (
            <span className="flex items-center gap-1.5">
              <Loader2 className="w-3.5 h-3.5 animate-spin" />
              {progressStatus ?? "Optimizing…"}
            </span>
          ) : result ? (
            "Re-optimize"
          ) : (
            "Optimize"
          )}
        </Button>
      </div>
    </div>
  );
}
