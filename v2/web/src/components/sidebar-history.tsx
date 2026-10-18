"use client";

import { useCallback } from "react";
import { Plus, Trash2 } from "lucide-react";
import { useOptimizerStore } from "@/store/optimizer";
import { Tooltip, TooltipContent, TooltipTrigger } from "@/components/ui/tooltip";

export function SidebarHistory() {
  const history = useOptimizerStore((s) => s.history);
  const activeId = useOptimizerStore((s) => s.activeId);
  const selectImage = useOptimizerStore((s) => s.selectImage);
  const addImage = useOptimizerStore((s) => s.addImage);
  const newImage = useOptimizerStore((s) => s.newImage);
  const removeImage = useOptimizerStore((s) => s.removeImage);

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
      // Paste image URL
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
      // Paste image from clipboard
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

  return (
    <div
      className="w-20 flex flex-col items-center gap-1 py-3 border-r border-gray-200 bg-gray-100/50 overflow-y-auto"
      onPaste={handlePaste}
    >
      {/* Add button: always returns to the blank drop-zone page, from which
          an image can be dropped, pasted, or chosen with "Choose file". */}
      <Tooltip>
        <TooltipTrigger
          className="w-14 h-14 rounded-xl border-2 border-dashed border-gray-300
            hover:border-gray-400 hover:bg-gray-200/50 text-gray-400
            inline-flex items-center justify-center shrink-0 cursor-pointer"
          onClick={newImage}
        >
          <Plus className="w-6 h-6" />
        </TooltipTrigger>
        <TooltipContent side="right">New image (drop or paste)</TooltipContent>
      </Tooltip>

      {/* History items */}
      {history.map((item) => (
        <div key={item.id} className="relative group">
          <button
            className={`w-14 h-14 rounded-xl overflow-hidden border-2 transition-all
              ${activeId === item.id
                ? "border-blue-500 shadow-md ring-1 ring-blue-300"
                : "border-gray-200 hover:border-gray-400"
              }`}
            onClick={() => selectImage(item.id)}
          >
            <img
              src={item.thumbnail}
              alt=""
              className="w-full h-full object-cover"
            />
          </button>
          {/* Status dot */}
          <div
            className={`absolute -top-1 -right-1 w-3 h-3 rounded-full border-2 border-white
              ${item.result ? "bg-green-500" : "bg-yellow-400"}`}
            title={item.result ? "Optimized" : "Pending"}
          />
          {/* Delete button */}
          <button
            className="absolute -bottom-1 -right-1 w-5 h-5 rounded-full bg-red-500 text-white
              flex items-center justify-center opacity-0 group-hover:opacity-100 transition-opacity
              hover:bg-red-600"
            onClick={(e) => {
              e.stopPropagation();
              removeImage(item.id);
            }}
          >
            <Trash2 className="w-3 h-3" />
          </button>
        </div>
      ))}
    </div>
  );
}
