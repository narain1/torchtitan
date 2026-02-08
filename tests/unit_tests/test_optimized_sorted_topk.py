# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import unittest

import torch

from torchtitan.models.moe.utils import _indices_dtype_by_sort_size


class TestOptimizedSortTopK(unittest.TestCase):
    """Test cases for optimized argsort and topk operations with indices_dtype."""

    def setUp(self):
        """Set up common test parameters."""
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        torch.manual_seed(42)

    def test_indices_dtype_by_sort_size(self):
        """Test that _indices_dtype_by_sort_size returns correct dtype based on size."""
        # Small sizes should use uint8
        self.assertEqual(_indices_dtype_by_sort_size(100), torch.uint8)
        self.assertEqual(_indices_dtype_by_sort_size(255), torch.uint8)
        
        # Medium sizes should use uint16
        self.assertEqual(_indices_dtype_by_sort_size(256), torch.uint16)
        self.assertEqual(_indices_dtype_by_sort_size(65535), torch.uint16)
        
        # Large sizes should use uint32
        self.assertEqual(_indices_dtype_by_sort_size(65536), torch.uint32)
        self.assertEqual(_indices_dtype_by_sort_size(4294967295), torch.uint32)
        
        # Very large sizes should use int64
        self.assertEqual(_indices_dtype_by_sort_size(4294967296), torch.int64)

    def test_optimized_argsort_correctness(self):
        """Test that optimized argsort produces correct results."""
        # Test with small tensor (uint8)
        data = torch.randint(0, 10, (100,), device=self.device)
        
        # Standard argsort
        standard_result = torch.argsort(data, stable=True)
        
        # Optimized argsort using out-variant
        indices_dtype = _indices_dtype_by_sort_size(data.numel())
        optimized_result = torch.empty(
            data.shape, dtype=indices_dtype, device=self.device
        )
        torch.argsort(data, stable=True, out=optimized_result)
        
        # Results should be the same
        self.assertTrue(torch.equal(standard_result, optimized_result.to(torch.int64)))
        
        # Verify the sorted data is correct
        self.assertTrue(
            torch.equal(data[standard_result], data[optimized_result.to(torch.int64)])
        )

    def test_optimized_argsort_large_tensor(self):
        """Test optimized argsort with larger tensor (uint16)."""
        data = torch.randint(0, 100, (10000,), device=self.device)
        
        # Standard argsort
        standard_result = torch.argsort(data, stable=True)
        
        # Optimized argsort using out-variant
        indices_dtype = _indices_dtype_by_sort_size(data.numel())
        self.assertEqual(indices_dtype, torch.uint16)
        
        optimized_result = torch.empty(
            data.shape, dtype=indices_dtype, device=self.device
        )
        torch.argsort(data, stable=True, out=optimized_result)
        
        # Results should be the same
        self.assertTrue(torch.equal(standard_result, optimized_result.to(torch.int64)))

    def test_optimized_topk_correctness(self):
        """Test that optimized topk produces correct results."""
        data = torch.rand(100, 50, device=self.device)
        k = 5
        
        # Standard topk
        standard_values, standard_indices = torch.topk(data, k=k, dim=-1, sorted=False)
        
        # Optimized topk using out-variant
        indices_dtype = _indices_dtype_by_sort_size(data.shape[-1])
        optimized_indices = torch.empty(
            data.shape[:-1] + (k,), dtype=indices_dtype, device=self.device
        )
        optimized_values = torch.empty(
            data.shape[:-1] + (k,), dtype=data.dtype, device=self.device
        )
        torch.topk(data, k=k, dim=-1, sorted=False, out=(optimized_values, optimized_indices))
        
        # Sort both results to compare (since topk with sorted=False can return in any order)
        standard_sorted_values, standard_sort_idx = torch.sort(standard_values, dim=-1)
        optimized_sorted_values, optimized_sort_idx = torch.sort(optimized_values, dim=-1)
        
        # Values should be the same
        self.assertTrue(torch.allclose(standard_sorted_values, optimized_sorted_values, rtol=1e-5))

    def test_optimized_operations_memory_dtype(self):
        """Test that optimized operations use smaller dtypes."""
        # Small tensor - should use uint8
        small_data = torch.randn(200, device=self.device)
        small_dtype = _indices_dtype_by_sort_size(small_data.numel())
        self.assertEqual(small_dtype, torch.uint8)
        
        # Verify uint8 uses less memory than int64
        small_indices = torch.empty(small_data.shape, dtype=small_dtype, device=self.device)
        large_indices = torch.empty(small_data.shape, dtype=torch.int64, device=self.device)
        self.assertLess(
            small_indices.element_size(),
            large_indices.element_size()
        )

    def test_integration_with_token_reorderer_size(self):
        """Test typical sizes used in TokenReorderer."""
        # Typical MoE scenario: batch_size * seq_len * top_k
        batch_size = 4
        seq_len = 2048
        top_k = 2
        num_experts = 8
        
        # Simulate expert indices
        expert_indices = torch.randint(
            0, num_experts, (batch_size * seq_len, top_k), device=self.device
        )
        flattened = expert_indices.view(-1)  # (4 * 2048 * 2) = 16384 elements
        
        # Check dtype selection
        indices_dtype = _indices_dtype_by_sort_size(flattened.numel())
        self.assertEqual(indices_dtype, torch.uint16)  # 16384 fits in uint16
        
        # Perform argsort with optimized dtype
        optimized_result = torch.empty(
            flattened.shape, dtype=indices_dtype, device=self.device
        )
        torch.argsort(flattened, stable=True, out=optimized_result)
        
        # Verify result is valid
        self.assertEqual(optimized_result.shape, flattened.shape)
        self.assertTrue((optimized_result >= 0).all())
        self.assertTrue((optimized_result < flattened.numel()).all())


if __name__ == "__main__":
    unittest.main()

