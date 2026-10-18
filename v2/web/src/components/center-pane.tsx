"use client";

import { useRef, useEffect, useCallback } from "react";
import { Download, Upload, ImageIcon, Plus } from "lucide-react";
import { Button } from "@/components/ui/button";
import { useOptimizerStore, useActiveItem } from "@/store/optimizer";
import { downloadZip } from "@/components/download-zip";
import { LayerView } from "@/components/layer-view";
import type { CenterTab } from "@/lib/types";

export function CenterPane() {
  const activeItem = useActiveItem();
  const centerTab = useOptimizerStore((s) => s.centerTab);
  const newImageMode = useOptimizerStore((s) => s.newImageMode);
  const setTab = useOptimizerStore((s) => s.setTab);
  const addImage = useOptimizerStore((s) => s.addImage);
  const newImage = useOptimizerStore((s) => s.newImage);
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const dropRef = useRef<HTMLDivElement>(null);

  const showDropZone = newImageMode || !activeItem;

  const tabs: { key: CenterTab; label: string; disabled?: boolean }[] = [
    { key: "input", label: "Input" },
    { key: "output", label: "Output", disabled: !activeItem?.result },
    { key: "layers", label: "Layers", disabled: !activeItem?.result },
  ];

  // Render image onto canvas
  const renderImage = useCallback((src: string) => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext("2d");
    if (!ctx) return;

    const img = new Image();
    img.onload = () => {
      const container = canvas.parentElement;
      const maxW = container?.clientWidth ?? 800;
      const maxH = container?.clientHeight ?? 600;
      const scale = Math.min(maxW / img.width, maxH / img.height);
      canvas.width = img.width * scale;
      canvas.height = img.height * scale;
      ctx.clearRect(0, 0, canvas.width, canvas.height);
      ctx.drawImage(img, 0, 0, canvas.width, canvas.height);
    };
    img.src = src;
  }, []);

  useEffect(() => {
    if (!activeItem) return;
    if (centerTab === "input") {
      renderImage(activeItem.imageDataUrl);
    } else if (centerTab === "output" && activeItem.result) {
      renderImage(`data:image/png;base64,${activeItem.result.mockup_base64}`);
    }
  }, [activeItem, centerTab, renderImage]);

  // Drag & drop on empty state
  const handleFile = useCallback(
    (file: File) => {
      if (!file.type.startsWith("image/")) return;
      const reader = new FileReader();
      reader.onload = () => addImage(reader.result as string);
      reader.readAsDataURL(file);
    },
    [addImage]
  );

  const handlePaste = useCallback(
    (e: React.ClipboardEvent) => {
      const text = e.clipboardData.getData("text/plain").trim();
      if (/^https?:\/\/.+\.(png|jpe?g|webp|gif|bmp)(\?.*)?$/i.test(text)) {
        e.preventDefault();
        fetch(text)
          .then((r) => r.blob())
          .then((blob) => {
            const reader = new FileReader();
            reader.onload = () => addImage(reader.result as string);
            reader.readAsDataURL(blob);
          })
          .catch(() => alert("Failed to load image from URL"));
        return;
      }
      const items = e.clipboardData.items;
      for (const item of items) {
        if (item.type.startsWith("image/")) {
          e.preventDefault();
          const file = item.getAsFile();
          if (file) handleFile(file);
          return;
        }
      }
    },
    [addImage, handleFile]
  );

  // Empty state (no image selected or explicit new-image mode)
  if (showDropZone) {
    return (
      <div
        ref={dropRef}
        className="flex-1 flex flex-col items-center justify-center gap-6 p-8 bg-gray-50 rounded-lg m-4"
        onPaste={handlePaste}
        onDrop={(e) => {
          e.preventDefault();
          const file = e.dataTransfer.files[0];
          if (file) handleFile(file);
        }}
        onDragOver={(e) => e.preventDefault()}
      >
        <div className="w-24 h-24 rounded-2xl bg-gray-200 flex items-center justify-center">
          <ImageIcon className="w-10 h-10 text-gray-400" />
        </div>
        <div className="text-center space-y-1">
          <p className="text-lg font-medium text-gray-500">
            Drop an image here or paste
          </p>
          <p className="text-sm text-gray-400">
            Supports PNG, JPEG, WebP — paste image URLs too
          </p>
        </div>
        <Button
          variant="outline"
          onClick={() => {
            const input = document.createElement("input");
            input.type = "file";
            input.accept = "image/*";
            input.onchange = () => {
              const file = input.files?.[0];
              if (file) handleFile(file);
            };
            input.click();
          }}
        >
          <Upload className="w-4 h-4 mr-2" />
          Choose file
        </Button>
      </div>
    );
  }

  return (
    <div className="flex-1 flex flex-col p-4 gap-3 min-w-0">
      {/* Tab bar */}
      <div className="flex items-center gap-1 border-b border-gray-200 pb-2">
        <div className="flex gap-0.5 flex-1">
          {tabs.map((tab) => (
            <button
              key={tab.key}
              disabled={tab.disabled}
              onClick={() => setTab(tab.key)}
              className={`px-3 py-1.5 text-sm rounded-md transition-colors
                ${centerTab === tab.key
                  ? "bg-gray-200 text-gray-800 font-medium"
                  : "text-gray-500 hover:text-gray-700 hover:bg-gray-100"
                }
                ${tab.disabled ? "opacity-40 cursor-not-allowed" : ""}
              `}
            >
              {tab.label}
            </button>
          ))}
        </div>

        {/* Download ZIP — always visible */}
        {activeItem.result && (
          <Button
            variant="ghost"
            size="sm"
            className="text-gray-500 hover:text-gray-700 gap-1.5"
            onClick={() => downloadZip(activeItem.result!)}
          >
            <Download className="w-3.5 h-3.5" />
            <span className="hidden sm:inline">Download ZIP</span>
          </Button>
        )}

        {/* + New button */}
        <Button
          variant="ghost"
          size="sm"
          className="text-gray-400 hover:text-gray-600 gap-1"
          onClick={newImage}
          title="New image"
        >
          <Plus className="w-3.5 h-3.5" />
        </Button>
      </div>

      {/* Content */}
      <div className="flex-1 flex items-center justify-center min-h-0">
        {centerTab === "layers" && activeItem.result ? (
          <LayerView result={activeItem.result} />
        ) : (
          <canvas
            ref={canvasRef}
            className="max-w-full max-h-full rounded-lg"
            style={{ imageRendering: "pixelated" }}
          />
        )}
      </div>
    </div>
  );
}
