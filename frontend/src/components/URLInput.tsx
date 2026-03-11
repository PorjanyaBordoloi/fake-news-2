import React, { useState, useRef } from 'react';
import { Plus, ImagePlus, X } from 'lucide-react';

interface Props {
    onSubmit: (url: string) => void;
    onSubmitImage?: (file: File) => void;
    isLoading: boolean;
}

export default function URLInput({ onSubmit, onSubmitImage, isLoading }: Props) {
    const [inputVal, setInputVal] = useState('');
    const [imageFile, setImageFile] = useState<File | null>(null);
    const fileInputRef = useRef<HTMLInputElement>(null);

    const handleSubmit = () => {
        if (isLoading) return;
        if (imageFile && onSubmitImage) {
            onSubmitImage(imageFile);
            setImageFile(null);
        } else if (inputVal.trim()) {
            onSubmit(inputVal.trim());
        }
    };

    const handleKeyDown = (e: React.KeyboardEvent) => {
        if (e.key === 'Enter' && !isLoading) handleSubmit();
    };

    const handleFileChange = (e: React.ChangeEvent<HTMLInputElement>) => {
        const file = e.target.files?.[0];
        if (file) {
            setImageFile(file);
            setInputVal('');
        }
        e.target.value = '';
    };

    return (
        <div className="input-section">
            {imageFile && (
                <div className="image-chip">
                    <span className="image-chip-name">📎 {imageFile.name}</span>
                    <button
                        className="image-chip-remove"
                        onClick={() => setImageFile(null)}
                        title="Remove image"
                    >
                        <X size={12} />
                    </button>
                </div>
            )}
            <div className="input-container">
                <Plus className="input-icon" onClick={handleSubmit} />
                <input
                    type="text"
                    className="url-input"
                    autoFocus
                    placeholder={imageFile
                        ? 'Press Enter or click + to analyze image...'
                        : 'Drop a news link or paste the article text here...'}
                    value={imageFile ? '' : inputVal}
                    onChange={(e) => { if (!imageFile) setInputVal(e.target.value); }}
                    onKeyDown={handleKeyDown}
                    disabled={isLoading}
                    readOnly={!!imageFile}
                />
                <button
                    className="image-upload-btn"
                    onClick={() => fileInputRef.current?.click()}
                    disabled={isLoading}
                    title="Upload an image to analyze"
                    type="button"
                >
                    <ImagePlus size={18} />
                </button>
                <input
                    ref={fileInputRef}
                    type="file"
                    accept="image/*"
                    style={{ display: 'none' }}
                    onChange={handleFileChange}
                />
            </div>
            <p className="input-hint">Unmask the truth: Paste a URL, article text, or upload an image to verify its credibility!</p>
        </div>
    );
}
