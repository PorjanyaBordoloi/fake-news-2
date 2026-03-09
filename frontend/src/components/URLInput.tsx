import React, { useState } from 'react';
import { Plus } from 'lucide-react';

interface Props {
    onSubmit: (url: string) => void;
    isLoading: boolean;
}

export default function URLInput({ onSubmit, isLoading }: Props) {
    const [inputVal, setInputVal] = useState('');

    const handleKeyDown = (e: React.KeyboardEvent) => {
        if (e.key === 'Enter' && inputVal.trim() && !isLoading) {
            onSubmit(inputVal.trim());
        }
    };

    return (
        <div className="input-section">
            <div className="input-container">
                <Plus className="input-icon" onClick={() => {
                    if (inputVal.trim() && !isLoading) onSubmit(inputVal.trim());
                }} />
                <input
                    type="text"
                    className="url-input"
                    autoFocus
                    placeholder="Paste News Article link here"
                    value={inputVal}
                    onChange={(e) => setInputVal(e.target.value)}
                    onKeyDown={handleKeyDown}
                    disabled={isLoading}
                />
            </div>
            <p className="input-hint">Paste News Article link here and verify CREDIBILITY</p>
        </div>
    );
}
