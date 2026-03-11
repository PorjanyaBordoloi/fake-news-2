import React, { useState, useRef } from 'react';
import { Plus, Image as ImageIcon, Loader2 } from 'lucide-react';

interface Props {
    onSubmit: (url: string) => void;
    onSubmitImage?: (file: File) => void;
    isLoading: boolean;
    isChatMode?: boolean;
}

export default function URLInput({ onSubmit, onSubmitImage, isLoading, isChatMode }: Props) {
    const [inputVal, setInputVal] = useState('');
    const [isUploading, setIsUploading] = useState(false);
    const fileInputRef = useRef<HTMLInputElement>(null);

    const handleKeyDown = (e: React.KeyboardEvent) => {
        if (e.key === 'Enter' && inputVal.trim() && !isLoading && !isUploading) {
            onSubmit(inputVal.trim());
            setInputVal('');
        }
    };

    const handleClickSubmit = () => {
        if (inputVal.trim() && !isLoading && !isUploading) {
            onSubmit(inputVal.trim());
            setInputVal('');
        }
    };

    const handleFileChange = async (e: React.ChangeEvent<HTMLInputElement>) => {
        const file = e.target.files?.[0];
        if (!file) return;

        // If consumer provided a direct image handler, use it for streaming
        if (onSubmitImage) {
            onSubmitImage(file);
            if (fileInputRef.current) fileInputRef.current.value = '';
            return;
        }

        // Otherwise fall back to OCR extract-then-submit flow
        setIsUploading(true);
        const formData = new FormData();
        formData.append('file', file);

        try {
            const apiUrl = import.meta.env.VITE_API_URL || 'http://localhost:8000';
            const response = await fetch(`${apiUrl}/api/extract-image`, {
                method: 'POST',
                body: formData,
            });

            if (response.ok) {
                const data = await response.json();
                if (data.extracted_text) {
                    onSubmit(data.extracted_text);
                    setInputVal('');
                } else {
                    alert('No text could be extracted from this image.');
                }
            } else {
                console.error('Image extraction failed:', await response.text());
                alert('Failed to analyze image.');
            }
        } catch (err) {
            console.error(err);
            alert('Error connecting to server.');
        } finally {
            setIsUploading(false);
            if (fileInputRef.current) {
                fileInputRef.current.value = '';
            }
        }
    };

    return (
        <div className={`input-section ${isChatMode ? 'fixed-bottom' : ''}`}>
            <div className="input-container">
                <input
                    type="file"
                    accept="image/*"
                    style={{ display: 'none' }}
                    ref={fileInputRef}
                    onChange={handleFileChange}
                />

                {isUploading ? (
                    <Loader2 className="input-icon spin" style={{ cursor: 'default' }} />
                ) : (
                    <ImageIcon
                        className="input-icon"
                        onClick={() => fileInputRef.current?.click()}
                    />
                )}

                <input
                    type="text"
                    className="url-input"
                    autoFocus
                    placeholder={isUploading ? 'Extracting text from image...' : 'Drop a news link or paste the article text here...'}
                    value={inputVal}
                    onChange={(e) => setInputVal(e.target.value)}
                    onKeyDown={handleKeyDown}
                    disabled={isLoading || isUploading}
                />
                <Plus className="input-icon submit-icon" onClick={handleClickSubmit} />
            </div>
            <p className={`input-hint ${isChatMode ? 'hidden' : ''}`}>Unmask the truth: Paste a URL, text, or upload an image to verify!</p>
        </div>
    );
}
