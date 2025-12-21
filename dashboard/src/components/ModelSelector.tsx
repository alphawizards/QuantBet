import { ChevronDown } from 'lucide-react';
import type { BettingModel } from '../types/api';

interface ModelSelectorProps {
    models: BettingModel[];
    selected: string;
    onSelect: (modelId: string) => void;
}

export function ModelSelector({ models, selected, onSelect }: ModelSelectorProps) {
    const selectedModel = models.find(m => m.id === selected);

    return (
        <div className="relative">
            <label className="block text-sm font-medium text-[var(--muted-foreground)] mb-2">
                Betting Model
            </label>

            <div className="relative">
                <select
                    value={selected}
                    onChange={(e) => onSelect(e.target.value)}
                    className="w-full appearance-none bg-[var(--card)] border border-[var(--border)] 
                     rounded-lg px-4 py-3 pr-10 text-[var(--foreground)] cursor-pointer
                     focus:outline-none focus:ring-2 focus:ring-[var(--primary)] focus:border-transparent
                     hover:border-[var(--primary)]/50 transition-colors"
                >
                    {models.map(model => (
                        <option key={model.id} value={model.id}>
                            {model.icon} {model.name}
                        </option>
                    ))}
                </select>

                <ChevronDown className="absolute right-3 top-1/2 -translate-y-1/2 w-5 h-5 text-[var(--muted-foreground)] pointer-events-none" />
            </div>

            {selectedModel && (
                <p className="mt-2 text-sm text-[var(--muted-foreground)]">
                    {selectedModel.description}
                </p>
            )}
        </div>
    );
}

interface ModelCardProps {
    model: BettingModel;
    isSelected: boolean;
    onClick: () => void;
}

export function ModelCard({ model, isSelected, onClick }: ModelCardProps) {
    return (
        <button
            onClick={onClick}
            className={`w-full text-left p-4 rounded-lg border transition-all ${isSelected
                    ? 'bg-[var(--primary)]/10 border-[var(--primary)] ring-2 ring-[var(--primary)]/30'
                    : 'bg-[var(--card)] border-[var(--border)] hover:border-[var(--primary)]/50'
                }`}
        >
            <div className="flex items-center gap-3">
                <span className="text-2xl">{model.icon}</span>
                <div>
                    <h4 className={`font-medium ${isSelected ? 'text-[var(--primary)]' : 'text-[var(--foreground)]'
                        }`}>
                        {model.name}
                    </h4>
                    <p className="text-sm text-[var(--muted-foreground)] mt-0.5">
                        {model.description}
                    </p>
                </div>
            </div>
        </button>
    );
}

interface ModelGridProps {
    models: BettingModel[];
    selected: string;
    onSelect: (modelId: string) => void;
}

export function ModelGrid({ models, selected, onSelect }: ModelGridProps) {
    return (
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-3">
            {models.map(model => (
                <ModelCard
                    key={model.id}
                    model={model}
                    isSelected={model.id === selected}
                    onClick={() => onSelect(model.id)}
                />
            ))}
        </div>
    );
}
