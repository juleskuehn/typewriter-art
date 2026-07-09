"use client";

import { useCallback, useRef, useState } from "react";
import { Upload, ClipboardPaste } from "lucide-react";
import { useOptimizerStore } from "@/store/optimizer";

export function ImageUpload() {
  const setFile = useOptimizerStore((s) => s.setFile);
  const [isDragOver, setIsDragOver] = useState(false);
  const fileInputRef = useRef<HTMLInputElement>(null);

  const handleFile = useCallback(
    (file: File) => {
      if (!file.type.startsWith("image/")) {
        alert("Please upload an image file (PNG, JPEG, etc.)");
        return;
      }
      setFile(file);
    },
    [setFile]
  );

  const handleDrop = useCallback(
    (e: React.DragEvent) => {
      e.preventDefault();
      setIsDragOver(false);
      const file = e.dataTransfer.files[0];
      if (file) handleFile(file);
    },
    [handleFile]
  );

  const handlePaste = useCallback(
    (e: React.ClipboardEvent) => {
      const items = e.clipboardData.items;
      for (const item of items) {
        if (item.type.startsWith("image/")) {
          const file = item.getAsFile();
          if (file) handleFile(file);
          return;
        }
      }
    },
    [handleFile]
  );

  return (
    <div
      className={`relative w-full max-w-sm aspect-square rounded-2xl border-2 border-dashed
        flex flex-col items-center justify-center gap-4 p-8 text-center
        transition-colors cursor-pointer
        ${isDragOver
          ? "border-blue-500 bg-blue-50 dark:bg-blue-950"
          : "border-gray-300 dark:border-gray-700 hover:border-gray-400 dark:hover:border-gray-600"
        }`}
      onDragOver={(e) => { e.preventDefault(); setIsDragOver(true); }}
      onDragLeave={() => setIsDragOver(false)}
      onDrop={handleDrop}
      onPaste={handlePaste}
      onClick={() => fileInputRef.current?.click()}
      tabIndex={0}
    >
      <Upload className="w-10 h-10 text-gray-400" />
      <div>
        <p className="text-sm font-medium text-gray-600 dark:text-gray-300">
          Drop an image here, paste, or click to browse
        </p>
        <p className="text-xs text-gray-400 mt-1">
          PNG, JPEG — grayscale photos work best
        </p>
      </div>
      <div className="flex items-center gap-1 text-xs text-gray-400">
        <ClipboardPaste className="w-3 h-3" />
        <span>Ctrl+V to paste from clipboard</span>
      </div>
      <input
        ref={fileInputRef}
        type="file"
        accept="image/*"
        className="hidden"
        onChange={(e) => {
          const file = e.target.files?.[0];
          if (file) handleFile(file);
        }}
      />
    </div>
  );
}
