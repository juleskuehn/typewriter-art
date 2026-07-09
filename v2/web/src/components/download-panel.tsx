"use client";

import { Button } from "@/components/ui/button";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Download, FileJson, Image } from "lucide-react";
import { useOptimizerStore } from "@/store/optimizer";
import { useCallback } from "react";

export function DownloadPanel() {
  const state = useOptimizerStore((s) => s.state);
  const result = state.phase === "complete" ? state.result : null;

  const downloadBlob = useCallback((base64: string, filename: string, mimeType: string) => {
    const byteChars = atob(base64);
    const byteNums = new Array(byteChars.length);
    for (let i = 0; i < byteChars.length; i++) {
      byteNums[i] = byteChars.charCodeAt(i);
    }
    const byteArr = new Uint8Array(byteNums);
    const blob = new Blob([byteArr], { type: mimeType });
    const url = URL.createObjectURL(blob);
    const a = document.createElement("a");
    a.href = url;
    a.download = filename;
    a.click();
    URL.revokeObjectURL(url);
  }, []);

  const downloadMockup = useCallback(() => {
    if (!result) return;
    downloadBlob(result.mockup_base64, "typewriter-art.png", "image/png");
  }, [result, downloadBlob]);

  const downloadChoices = useCallback(() => {
    if (!result) return;
    const json = JSON.stringify(result.choices, null, 2);
    const blob = new Blob([json], { type: "application/json" });
    const url = URL.createObjectURL(blob);
    const a = document.createElement("a");
    a.href = url;
    a.download = "choices.json";
    a.click();
    URL.revokeObjectURL(url);
  }, [result]);

  const downloadLayer = useCallback(
    (index: number) => {
      if (!result) return;
      const base64 = result.layers_base64[index];
      const [y, x] = result.fractional_offsets[index];
      downloadBlob(base64, `layer_${index}_${y}_${x}.png`, "image/png");
    },
    [result, downloadBlob]
  );

  if (!result) return null;

  return (
    <Card className="w-full max-w-sm">
      <CardHeader>
        <CardTitle className="text-lg flex items-center gap-2">
          <Download className="w-4 h-4" />
          Download
        </CardTitle>
      </CardHeader>
      <CardContent className="space-y-2">
        <Button
          variant="outline"
          className="w-full justify-start gap-2"
          onClick={downloadMockup}
        >
          <Image className="w-4 h-4" />
          Final mockup (PNG)
        </Button>

        <Button
          variant="outline"
          className="w-full justify-start gap-2"
          onClick={downloadChoices}
        >
          <FileJson className="w-4 h-4" />
          Choices (JSON)
        </Button>

        <div className="pt-2 border-t border-gray-200 dark:border-gray-800">
          <p className="text-xs text-gray-500 mb-2">Individual layers:</p>
          <div className="grid grid-cols-2 gap-1">
            {result.layers_base64.map((_, i) => (
              <Button
                key={i}
                variant="ghost"
                size="sm"
                className="justify-start text-xs"
                onClick={() => downloadLayer(i)}
              >
                <Image className="w-3 h-3 mr-1" />
                Layer {i + 1}
              </Button>
            ))}
          </div>
        </div>
      </CardContent>
    </Card>
  );
}
