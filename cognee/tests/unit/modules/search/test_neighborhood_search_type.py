import pytest

from cognee.modules.retrieval.neighborhood_retriever import NeighborhoodRetriever
from cognee.modules.search.methods.get_search_type_retriever_instance import (
    get_search_type_retriever_instance,
)
from cognee.modules.search.types import SearchType


@pytest.mark.asyncio
async def test_neighborhood_search_type_registry():
    retriever = await get_search_type_retriever_instance(
        query_type=SearchType.NEIGHBORHOOD,
        query_text="Acme",
        retriever_specific_config={
            "depth": 2,
            "seed_top_k": 3,
            "edge_types": ["works_at"],
        },
    )

    assert isinstance(retriever, NeighborhoodRetriever)
    assert retriever.neighborhood_depth == 2
    assert retriever.neighborhood_seed_top_k == 3
    assert retriever.edge_types == ["works_at"]
