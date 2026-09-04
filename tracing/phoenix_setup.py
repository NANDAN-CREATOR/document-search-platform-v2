"""
Full Arize Phoenix tracing setup.
On Docker: exports to Phoenix container at http://phoenix:4317
"""
import logging

logger = logging.getLogger(__name__)


def instrument_llamaindex() -> None:
    """Instrument LlamaIndex with full Arize Phoenix tracing."""
    try:
        from opentelemetry import trace
        from opentelemetry.sdk.trace import TracerProvider
        from opentelemetry.sdk.trace.export import BatchSpanProcessor
        from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter
        from openinference.instrumentation.llama_index import LlamaIndexInstrumentor
        from config.settings import settings

        otlp_exporter = OTLPSpanExporter(
            endpoint=f"http://{settings.phoenix_host}:{settings.phoenix_grpc_port}",
            insecure=True,
        )
        provider = TracerProvider()
        provider.add_span_processor(BatchSpanProcessor(otlp_exporter))
        trace.set_tracer_provider(provider)
        LlamaIndexInstrumentor().instrument(tracer_provider=provider)
        logger.info(f"Arize Phoenix tracing enabled → http://{settings.phoenix_host}:{settings.phoenix_port}")
    except Exception as e:
        logger.warning(f"Phoenix tracing unavailable: {e}")


def instrument_all() -> None:
    instrument_llamaindex()
