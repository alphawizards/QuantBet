import { Component } from 'react';
import type { ErrorInfo, ReactNode } from 'react';
import { AlertTriangle, RefreshCw, Home } from 'lucide-react';

/**
 * Props for ErrorBoundary component.
 */
interface ErrorBoundaryProps {
    children: ReactNode;
    fallback?: ReactNode;
    onError?: (error: Error, errorInfo: ErrorInfo) => void;
}

/**
 * State for ErrorBoundary component.
 */
interface ErrorBoundaryState {
    hasError: boolean;
    error: Error | null;
    errorInfo: ErrorInfo | null;
}

/**
 * ErrorBoundary Component
 * 
 * Catches JavaScript errors anywhere in the child component tree,
 * logs the errors, and displays a fallback UI instead of crashing.
 * 
 * Features:
 * - Customizable fallback UI
 * - Error logging callback
 * - Retry functionality
 * - Navigation to home
 * 
 * @example
 * <ErrorBoundary fallback={<CustomError />}>
 *   <ChildComponent />
 * </ErrorBoundary>
 */
export class ErrorBoundary extends Component<ErrorBoundaryProps, ErrorBoundaryState> {
    constructor(props: ErrorBoundaryProps) {
        super(props);
        this.state = {
            hasError: false,
            error: null,
            errorInfo: null
        };
    }

    static getDerivedStateFromError(error: Error): Partial<ErrorBoundaryState> {
        // Update state so the next render will show the fallback UI.
        return { hasError: true, error };
    }

    componentDidCatch(error: Error, errorInfo: ErrorInfo): void {
        // Log error to console in development
        console.error('ErrorBoundary caught an error:', error, errorInfo);

        // Store error info for display
        this.setState({ errorInfo });

        // Call optional error callback
        if (this.props.onError) {
            this.props.onError(error, errorInfo);
        }

        // In production, you might want to log to an error tracking service
        // Example: Sentry.captureException(error, { extra: errorInfo });
    }

    handleRetry = (): void => {
        this.setState({
            hasError: false,
            error: null,
            errorInfo: null
        });
    };

    handleGoHome = (): void => {
        window.location.href = '/';
    };

    render(): ReactNode {
        if (this.state.hasError) {
            // If a custom fallback is provided, use it
            if (this.props.fallback) {
                return this.props.fallback;
            }

            // Default fallback UI
            return (
                <div className="error-boundary">
                    <div className="error-boundary-content">
                        <div className="error-icon">
                            <AlertTriangle size={64} color="#ef4444" />
                        </div>

                        <h1 className="error-title">Something went wrong</h1>

                        <p className="error-message">
                            We're sorry, but something unexpected happened.
                            Please try refreshing the page or return to the home page.
                        </p>

                        {import.meta.env.DEV && this.state.error && (
                            <details className="error-details">
                                <summary>Error Details (Development Only)</summary>
                                <pre className="error-stack">
                                    {this.state.error.toString()}
                                    {this.state.errorInfo?.componentStack}
                                </pre>
                            </details>
                        )}

                        <div className="error-actions">
                            <button
                                className="error-button error-button-primary"
                                onClick={this.handleRetry}
                            >
                                <RefreshCw size={16} />
                                Try Again
                            </button>

                            <button
                                className="error-button error-button-secondary"
                                onClick={this.handleGoHome}
                            >
                                <Home size={16} />
                                Go Home
                            </button>
                        </div>
                    </div>

                    <style>{`
                        .error-boundary {
                            min-height: 100vh;
                            display: flex;
                            align-items: center;
                            justify-content: center;
                            background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%);
                            padding: 2rem;
                        }

                        .error-boundary-content {
                            max-width: 500px;
                            text-align: center;
                            background: rgba(255, 255, 255, 0.05);
                            border: 1px solid rgba(255, 255, 255, 0.1);
                            border-radius: 16px;
                            padding: 3rem;
                            backdrop-filter: blur(10px);
                        }

                        .error-icon {
                            margin-bottom: 1.5rem;
                            animation: pulse 2s infinite;
                        }

                        @keyframes pulse {
                            0%, 100% { opacity: 1; }
                            50% { opacity: 0.5; }
                        }

                        .error-title {
                            font-size: 1.75rem;
                            font-weight: 700;
                            color: #ffffff;
                            margin-bottom: 1rem;
                        }

                        .error-message {
                            color: rgba(255, 255, 255, 0.7);
                            line-height: 1.6;
                            margin-bottom: 2rem;
                        }

                        .error-details {
                            text-align: left;
                            background: rgba(0, 0, 0, 0.3);
                            border-radius: 8px;
                            padding: 1rem;
                            margin-bottom: 2rem;
                            cursor: pointer;
                        }

                        .error-details summary {
                            color: rgba(255, 255, 255, 0.6);
                            font-size: 0.875rem;
                            margin-bottom: 0.5rem;
                        }

                        .error-stack {
                            color: #ef4444;
                            font-size: 0.75rem;
                            overflow-x: auto;
                            white-space: pre-wrap;
                            word-break: break-all;
                            margin: 0;
                            padding-top: 0.5rem;
                        }

                        .error-actions {
                            display: flex;
                            gap: 1rem;
                            justify-content: center;
                        }

                        .error-button {
                            display: inline-flex;
                            align-items: center;
                            gap: 0.5rem;
                            padding: 0.75rem 1.5rem;
                            font-size: 0.875rem;
                            font-weight: 600;
                            border-radius: 8px;
                            cursor: pointer;
                            transition: all 0.2s ease;
                            border: none;
                        }

                        .error-button-primary {
                            background: linear-gradient(135deg, #6366f1 0%, #8b5cf6 100%);
                            color: white;
                        }

                        .error-button-primary:hover {
                            transform: translateY(-2px);
                            box-shadow: 0 4px 12px rgba(99, 102, 241, 0.4);
                        }

                        .error-button-secondary {
                            background: rgba(255, 255, 255, 0.1);
                            color: white;
                            border: 1px solid rgba(255, 255, 255, 0.2);
                        }

                        .error-button-secondary:hover {
                            background: rgba(255, 255, 255, 0.15);
                        }
                    `}</style>
                </div>
            );
        }

        return this.props.children;
    }
}

/**
 * useErrorBoundary Hook
 * 
 * For functional components, provides a way to trigger error boundary behavior.
 * Useful for async errors that aren't caught by componentDidCatch.
 * 
 * @example
 * const { showBoundary } = useErrorBoundary();
 * 
 * try {
 *   await fetchData();
 * } catch (error) {
 *   showBoundary(error);
 * }
 */
export function withErrorBoundary<P extends object>(
    WrappedComponent: React.ComponentType<P>,
    fallback?: ReactNode
): React.FC<P> {
    return function WithErrorBoundaryWrapper(props: P) {
        return (
            <ErrorBoundary fallback={fallback}>
                <WrappedComponent {...props} />
            </ErrorBoundary>
        );
    };
}

export default ErrorBoundary;
