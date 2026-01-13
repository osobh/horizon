//! Page Table Implementation
//!
//! Virtual to physical page mapping with LRU eviction and dirty tracking

use super::TierLevel;
use anyhow::{Context, Result};
use std::collections::{HashMap, VecDeque};
use std::time::Instant;

/// Page identifier
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct PageId(u64);

impl PageId {
    /// Create new page ID
    pub fn new(id: u64) -> Self {
        Self(id)
    }

    /// Get ID as u64
    pub fn as_u64(&self) -> u64 {
        self.0
    }

    /// Check if valid (non-zero)
    pub fn is_valid(&self) -> bool {
        self.0 != 0
    }
}

/// Page information
#[derive(Debug, Clone)]
pub struct PageInfo {
    pub tier: TierLevel,
    pub address: u64,
    pub dirty: bool,
    pub access_count: u64,
    pub last_access: Instant,
}

/// Page table with LRU eviction
pub struct PageTable {
    /// Page ID to info mapping
    pages: HashMap<PageId, PageInfo>,

    /// LRU queue of page IDs
    lru_queue: VecDeque<PageId>,

    /// Maximum pages before eviction
    max_pages: usize,

    /// Statistics
    hits: u64,
    misses: u64,
    evictions: u64,
}

impl PageTable {
    /// Create new page table
    pub fn new(max_pages: usize) -> Self {
        Self {
            pages: HashMap::with_capacity(max_pages),
            lru_queue: VecDeque::with_capacity(max_pages),
            max_pages,
            hits: 0,
            misses: 0,
            evictions: 0,
        }
    }

    /// Insert page into table
    pub fn insert(&mut self, page_id: PageId, mut info: PageInfo) -> Result<Option<PageInfo>> {
        // Check if we need to evict
        if self.pages.len() >= self.max_pages && !self.pages.contains_key(&page_id) {
            // Evict LRU page
            if let Some(evict_id) = self.lru_queue.pop_front() {
                if let Some(evicted) = self.pages.remove(&evict_id) {
                    self.evictions += 1;
                    return Ok(Some(evicted));
                }
            }
        }

        // Update access time
        info.last_access = Instant::now();

        // Insert or update
        self.pages.insert(page_id, info);

        // Update LRU queue
        self.update_lru(page_id);

        Ok(None)
    }

    /// Lookup page
    pub fn lookup(&self, page_id: PageId) -> Option<&PageInfo> {
        let info = self.pages.get(&page_id);

        if info.is_some() {
            // Note: Can't update LRU here due to &self
            // Use access_page() for proper LRU update
        }

        info
    }

    /// Lookup page mutably
    pub fn lookup_mut(&mut self, page_id: PageId) -> Option<&mut PageInfo> {
        self.pages.get_mut(&page_id)
    }

    /// Access page (updates LRU and stats)
    pub fn access_page(&mut self, page_id: PageId) {
        if let Some(info) = self.pages.get_mut(&page_id) {
            info.access_count += 1;
            info.last_access = Instant::now();
            self.hits += 1;
            self.update_lru(page_id);
        } else {
            self.misses += 1;
        }
    }

    /// Remove page from table
    pub fn remove(&mut self, page_id: PageId) -> Result<PageInfo> {
        // Remove from LRU queue
        self.lru_queue.retain(|&id| id != page_id);

        // Remove from pages
        self.pages
            .remove(&page_id)
            .context("Page not found in table")
    }

    /// Mark page as dirty
    pub fn mark_dirty(&mut self, page_id: PageId) -> Result<()> {
        let info = self.pages.get_mut(&page_id).context("Page not found")?;
        info.dirty = true;
        Ok(())
    }

    /// Get all dirty pages
    pub fn get_dirty_pages(&self) -> Vec<(PageId, &PageInfo)> {
        self.pages
            .iter()
            .filter(|(_, info)| info.dirty)
            .map(|(id, info)| (*id, info))
            .collect()
    }

    /// Clear dirty flag
    pub fn clear_dirty(&mut self, page_id: PageId) -> Result<()> {
        let info = self.pages.get_mut(&page_id).context("Page not found")?;
        info.dirty = false;
        Ok(())
    }

    /// Get LRU candidates for eviction
    pub fn get_eviction_candidates(&self, count: usize) -> Vec<PageId> {
        self.lru_queue.iter().take(count).copied().collect()
    }

    /// Get page count
    pub fn page_count(&self) -> usize {
        self.pages.len()
    }

    /// Iterate over all pages
    pub fn iter_pages(&self) -> impl Iterator<Item = (&PageId, &PageInfo)> {
        self.pages.iter()
    }

    /// Get statistics
    pub fn get_stats(&self) -> PageTableStats {
        PageTableStats {
            total_pages: self.pages.len(),
            hits: self.hits,
            misses: self.misses,
            evictions: self.evictions,
            hit_rate: if self.hits + self.misses > 0 {
                self.hits as f64 / (self.hits + self.misses) as f64
            } else {
                0.0
            },
        }
    }

    /// Update LRU queue
    fn update_lru(&mut self, page_id: PageId) {
        // Remove from current position
        self.lru_queue.retain(|&id| id != page_id);

        // Add to back (most recently used)
        self.lru_queue.push_back(page_id);
    }
}

/// Page table statistics
#[derive(Debug)]
pub struct PageTableStats {
    pub total_pages: usize,
    pub hits: u64,
    pub misses: u64,
    pub evictions: u64,
    pub hit_rate: f64,
}

/// Multi-level page table for large address spaces
pub struct MultiLevelPageTable {
    /// Root level tables
    root: HashMap<u64, Box<PageTableLevel>>,

    /// Bits per level
    bits_per_level: usize,

    /// Number of levels
    num_levels: usize,
}

/// Page table level
struct PageTableLevel {
    entries: HashMap<u64, PageTableEntry>,
}

/// Page table entry
enum PageTableEntry {
    Page(PageInfo),
    Table(Box<PageTableLevel>),
}

impl MultiLevelPageTable {
    /// Create new multi-level page table
    pub fn new(bits_per_level: usize, num_levels: usize) -> Self {
        Self {
            root: HashMap::new(),
            bits_per_level,
            num_levels,
        }
    }

    /// Insert page
    pub fn insert(&mut self, address: u64, info: PageInfo) -> Result<()> {
        let indices = self.split_address(address);

        // Navigate to correct level
        let mut current_level = self.root.entry(indices[0]).or_insert_with(|| {
            Box::new(PageTableLevel {
                entries: HashMap::new(),
            })
        });

        // Navigate through intermediate levels
        for i in 1..self.num_levels - 1 {
            let entry = current_level.entries.entry(indices[i]).or_insert_with(|| {
                PageTableEntry::Table(Box::new(PageTableLevel {
                    entries: HashMap::new(),
                }))
            });

            if let PageTableEntry::Table(table) = entry {
                current_level = table;
            } else {
                anyhow::bail!("Invalid page table structure");
            }
        }

        // Insert at leaf level
        current_level
            .entries
            .insert(indices[self.num_levels - 1], PageTableEntry::Page(info));

        Ok(())
    }

    /// Lookup page
    pub fn lookup(&self, address: u64) -> Option<&PageInfo> {
        let indices = self.split_address(address);

        // Navigate to correct level
        let mut current_level = self.root.get(&indices[0])?;

        // Navigate through intermediate levels
        for i in 1..self.num_levels - 1 {
            if let Some(PageTableEntry::Table(table)) = current_level.entries.get(&indices[i]) {
                current_level = table;
            } else {
                return None;
            }
        }

        // Get from leaf level
        if let Some(PageTableEntry::Page(info)) =
            current_level.entries.get(&indices[self.num_levels - 1])
        {
            Some(info)
        } else {
            None
        }
    }

    /// Split address into level indices
    fn split_address(&self, address: u64) -> Vec<u64> {
        let mut indices = Vec::with_capacity(self.num_levels);
        let mask = (1u64 << self.bits_per_level) - 1;

        for i in 0..self.num_levels {
            let shift = (self.num_levels - i - 1) * self.bits_per_level;
            indices.push((address >> shift) & mask);
        }

        indices
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_page_table_basic() {
        let mut pt = PageTable::new(100);

        let page_id = PageId::new(1);
        let info = PageInfo {
            tier: TierLevel::Gpu,
            address: 0x1000,
            dirty: false,
            access_count: 0,
            last_access: Instant::now(),
        };

        pt.insert(page_id, info)?;
        assert!(pt.lookup(page_id).is_some());

        pt.access_page(page_id);
        let stats = pt.get_stats();
        assert_eq!(stats.hits, 1);
        assert_eq!(stats.misses, 0);
    }

    #[test]
    fn test_lru_eviction() {
        let mut pt = PageTable::new(3);

        // Insert 3 pages
        for i in 1..=3 {
            let page_id = PageId::new(i);
            let info = PageInfo {
                tier: TierLevel::Cpu,
                address: i as u64 * 0x1000,
                dirty: false,
                access_count: 0,
                last_access: Instant::now(),
            };
            pt.insert(page_id, info)?;
        }

        // Access pages 1 and 2
        pt.access_page(PageId::new(1));
        pt.access_page(PageId::new(2));

        // Insert page 4 - should evict page 3
        let page4 = PageId::new(4);
        let info4 = PageInfo {
            tier: TierLevel::Cpu,
            address: 0x4000,
            dirty: false,
            access_count: 0,
            last_access: Instant::now(),
        };

        let evicted = pt.insert(page4, info4)?;
        assert!(evicted.is_some());

        // Page 3 should be evicted
        assert!(pt.lookup(PageId::new(3)).is_none());
        assert!(pt.lookup(PageId::new(1)).is_some());
        assert!(pt.lookup(PageId::new(2)).is_some());
        assert!(pt.lookup(PageId::new(4)).is_some());
    }
}
