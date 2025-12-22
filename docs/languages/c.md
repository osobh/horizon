# C Kernel Development Rules

## Overview

This document defines comprehensive coding standards, length limits, linting configurations, and Test-Driven Development (TDD) practices for C-based kernel development on Ubuntu 25.04. All development follows a strict TDD approach with kernel-specific testing frameworks.

## C Coding Standards for Kernel Development

### Style Guide and Formatting

- **Style Guide**: Linux Kernel Coding Style + GNU C Standards
- **Formatter**: clang-format with kernel configuration
- **Line Length**: 80 characters (kernel standard)
- **Indentation**: 8-character tabs (kernel standard)
- **Naming**: Follow Linux kernel naming conventions strictly
- **C Standard**: C11 with GNU extensions (kernel-compatible)

### Include Organization

- Group includes: System headers, kernel headers, local headers
- Use include guards for all headers
- Minimize header dependencies
- Sort includes within groups
- Use angle brackets for system/kernel, quotes for local

```c
/* System headers */
#include <linux/init.h>
#include <linux/module.h>
#include <linux/kernel.h>
#include <linux/slab.h>
#include <linux/fs.h>
#include <linux/errno.h>
#include <linux/types.h>
#include <linux/cdev.h>
#include <linux/device.h>
#include <linux/mutex.h>
#include <linux/spinlock.h>

/* Architecture-specific headers */
#include <asm/uaccess.h>
#include <asm/io.h>

/* Local module headers */
#include "driver_core.h"
#include "driver_ioctl.h"
#include "driver_debug.h"
```

### Naming Conventions

| Element              | Convention           | Example                    |
| -------------------- | -------------------- | -------------------------- |
| Files                | lowercase_underscore | `driver_core.c`            |
| Functions            | lowercase_underscore | `init_driver_module`       |
| Global Variables     | lowercase_underscore | `driver_major_number`      |
| Local Variables      | lowercase_underscore | `buffer_size`              |
| Constants            | UPPERCASE_UNDERSCORE | `MAX_BUFFER_SIZE`          |
| Macros               | UPPERCASE_UNDERSCORE | `DRIVER_IOCTL_MAGIC`       |
| Structs              | lowercase_underscore | `struct driver_data`       |
| Struct Members       | lowercase_underscore | `data->buffer_ptr`         |
| Enums                | lowercase_underscore | `enum driver_state`        |
| Enum Values          | UPPERCASE_UNDERSCORE | `DRIVER_STATE_READY`       |
| Typedef              | lowercase_underscore_t | `driver_handle_t`        |
| Function Pointers    | lowercase_underscore | `(*read_handler)`          |

### Kernel-Specific Standards

#### Memory Management

- Always check kmalloc/kzalloc return values
- Use appropriate GFP flags for allocation context
- Free all allocated memory in error paths
- Use kernel memory debugging tools (KASAN, kmemleak)
- Prefer stack allocation for small, fixed-size data

```c
/* Good: Proper kernel memory allocation with error handling */
struct driver_data *alloc_driver_data(size_t buffer_size)
{
	struct driver_data *data;
	
	/* Allocate main structure */
	data = kzalloc(sizeof(*data), GFP_KERNEL);
	if (!data)
		return ERR_PTR(-ENOMEM);
	
	/* Initialize mutex before any potential failure */
	mutex_init(&data->lock);
	spin_lock_init(&data->spinlock);
	
	/* Allocate buffer with size validation */
	if (buffer_size > MAX_BUFFER_SIZE) {
		kfree(data);
		return ERR_PTR(-EINVAL);
	}
	
	data->buffer = kmalloc(buffer_size, GFP_KERNEL);
	if (!data->buffer) {
		kfree(data);
		return ERR_PTR(-ENOMEM);
	}
	
	data->buffer_size = buffer_size;
	atomic_set(&data->ref_count, 1);
	
	return data;
}

/* Good: Proper cleanup with reference counting */
void free_driver_data(struct driver_data *data)
{
	if (!data || IS_ERR(data))
		return;
	
	if (atomic_dec_and_test(&data->ref_count)) {
		mutex_destroy(&data->lock);
		kfree(data->buffer);
		kfree(data);
	}
}

/* Good: DMA-aware memory allocation */
static int alloc_dma_buffers(struct device *dev, struct dma_buffer *buf)
{
	dma_addr_t dma_handle;
	
	buf->cpu_addr = dma_alloc_coherent(dev, DMA_BUFFER_SIZE,
					    &dma_handle, GFP_KERNEL);
	if (!buf->cpu_addr) {
		dev_err(dev, "Failed to allocate DMA buffer\n");
		return -ENOMEM;
	}
	
	buf->dma_addr = dma_handle;
	buf->size = DMA_BUFFER_SIZE;
	
	return 0;
}
```

#### Error Handling

- Use kernel error codes (negative errno values)
- Propagate errors up the call stack
- Clean up resources in reverse order on error
- Use goto for error handling in complex functions
- Log errors appropriately with pr_* or dev_*

```c
/* Good: Kernel-style error handling with goto */
static int driver_probe(struct platform_device *pdev)
{
	struct driver_data *drvdata;
	struct resource *res;
	int ret;
	
	/* Allocate driver data */
	drvdata = devm_kzalloc(&pdev->dev, sizeof(*drvdata), GFP_KERNEL);
	if (!drvdata)
		return -ENOMEM;
	
	/* Get and map IO resources */
	res = platform_get_resource(pdev, IORESOURCE_MEM, 0);
	drvdata->base = devm_ioremap_resource(&pdev->dev, res);
	if (IS_ERR(drvdata->base))
		return PTR_ERR(drvdata->base);
	
	/* Get IRQ */
	drvdata->irq = platform_get_irq(pdev, 0);
	if (drvdata->irq < 0)
		return drvdata->irq;
	
	/* Initialize hardware */
	ret = driver_hw_init(drvdata);
	if (ret) {
		dev_err(&pdev->dev, "Hardware init failed: %d\n", ret);
		goto err_hw_init;
	}
	
	/* Request IRQ */
	ret = devm_request_irq(&pdev->dev, drvdata->irq, driver_irq_handler,
			       IRQF_SHARED, dev_name(&pdev->dev), drvdata);
	if (ret) {
		dev_err(&pdev->dev, "Failed to request IRQ: %d\n", ret);
		goto err_irq;
	}
	
	/* Register character device */
	ret = driver_register_cdev(drvdata);
	if (ret) {
		dev_err(&pdev->dev, "Failed to register cdev: %d\n", ret);
		goto err_cdev;
	}
	
	platform_set_drvdata(pdev, drvdata);
	dev_info(&pdev->dev, "Driver probed successfully\n");
	
	return 0;
	
err_cdev:
	/* IRQ freed automatically by devm */
err_irq:
	driver_hw_cleanup(drvdata);
err_hw_init:
	/* Memory freed automatically by devm */
	return ret;
}

/* Good: Inline error checking for critical paths */
static inline int validate_user_buffer(const void __user *buf, size_t len)
{
	if (!buf || !len)
		return -EINVAL;
	
	if (len > MAX_USER_BUFFER_SIZE)
		return -E2BIG;
	
	if (!access_ok(buf, len))
		return -EFAULT;
	
	return 0;
}
```

#### Locking and Synchronization

- Document all locking requirements
- Use appropriate locking primitives (mutex, spinlock, RCU)
- Avoid nested locking when possible
- Use lockdep annotations
- Follow locking hierarchy to prevent deadlocks

```c
/* Good: Documented locking with proper nesting */

/**
 * struct driver_device - Main device structure
 * @lock: Protects @buffer and @buffer_size (mutex)
 * @hw_lock: Protects hardware registers (spinlock, IRQ-safe)
 * @list_lock: Protects @pending_list (spinlock)
 * 
 * Lock ordering: @lock -> @list_lock -> @hw_lock
 */
struct driver_device {
	struct mutex lock;
	spinlock_t hw_lock;
	spinlock_t list_lock;
	
	void *buffer;
	size_t buffer_size;
	
	void __iomem *regs;
	struct list_head pending_list;
	
	atomic_t pending_count;
	wait_queue_head_t wait_queue;
};

/* Good: Proper spinlock usage in IRQ context */
static irqreturn_t driver_irq_handler(int irq, void *dev_id)
{
	struct driver_device *dev = dev_id;
	unsigned long flags;
	u32 status;
	
	spin_lock_irqsave(&dev->hw_lock, flags);
	
	status = readl(dev->regs + DRIVER_STATUS_REG);
	if (!(status & DRIVER_IRQ_PENDING)) {
		spin_unlock_irqrestore(&dev->hw_lock, flags);
		return IRQ_NONE;
	}
	
	/* Clear interrupt */
	writel(status, dev->regs + DRIVER_STATUS_REG);
	
	spin_unlock_irqrestore(&dev->hw_lock, flags);
	
	/* Process in bottom half */
	if (status & DRIVER_DATA_READY) {
		atomic_inc(&dev->pending_count);
		wake_up_interruptible(&dev->wait_queue);
	}
	
	return IRQ_HANDLED;
}

/* Good: RCU for read-heavy data structures */
struct driver_config {
	struct rcu_head rcu;
	int param1;
	int param2;
};

static void update_config(struct driver_device *dev, int p1, int p2)
{
	struct driver_config *old_cfg, *new_cfg;
	
	new_cfg = kmalloc(sizeof(*new_cfg), GFP_KERNEL);
	if (!new_cfg)
		return;
	
	new_cfg->param1 = p1;
	new_cfg->param2 = p2;
	
	mutex_lock(&dev->lock);
	old_cfg = rcu_dereference_protected(dev->config,
					    lockdep_is_held(&dev->lock));
	rcu_assign_pointer(dev->config, new_cfg);
	mutex_unlock(&dev->lock);
	
	if (old_cfg)
		kfree_rcu(old_cfg, rcu);
}
```

#### Kernel API Usage

- Use appropriate kernel subsystem APIs
- Follow subsystem-specific conventions
- Handle all possible return values
- Use kernel data structures (list, rbtree, etc.)

```c
/* Good: Proper device model integration */
static int driver_create_sysfs_attrs(struct device *dev)
{
	int ret;
	
	ret = device_create_file(dev, &dev_attr_buffer_size);
	if (ret) {
		dev_err(dev, "Failed to create buffer_size attr\n");
		return ret;
	}
	
	ret = device_create_file(dev, &dev_attr_status);
	if (ret) {
		dev_err(dev, "Failed to create status attr\n");
		goto err_status;
	}
	
	ret = sysfs_create_group(&dev->kobj, &driver_attr_group);
	if (ret) {
		dev_err(dev, "Failed to create attr group\n");
		goto err_group;
	}
	
	return 0;
	
err_group:
	device_remove_file(dev, &dev_attr_status);
err_status:
	device_remove_file(dev, &dev_attr_buffer_size);
	return ret;
}

/* Good: Workqueue usage for deferred work */
static void driver_work_handler(struct work_struct *work)
{
	struct driver_work *dwork = container_of(work, struct driver_work,
						 work.work);
	struct driver_device *dev = dwork->dev;
	unsigned long flags;
	
	mutex_lock(&dev->lock);
	
	/* Process pending items */
	spin_lock_irqsave(&dev->list_lock, flags);
	while (!list_empty(&dev->pending_list)) {
		struct driver_request *req;
		
		req = list_first_entry(&dev->pending_list,
				      struct driver_request, list);
		list_del_init(&req->list);
		spin_unlock_irqrestore(&dev->list_lock, flags);
		
		/* Process request without lock held */
		driver_process_request(dev, req);
		
		spin_lock_irqsave(&dev->list_lock, flags);
	}
	spin_unlock_irqrestore(&dev->list_lock, flags);
	
	mutex_unlock(&dev->lock);
	
	/* Reschedule if more work pending */
	if (atomic_read(&dev->pending_count) > 0)
		queue_delayed_work(system_wq, &dwork->work,
				  msecs_to_jiffies(10));
}
```

### Documentation Standards

#### Kernel-Doc Comments

```c
/**
 * struct driver_buffer - DMA buffer descriptor
 * @cpu_addr: CPU virtual address of the buffer
 * @dma_addr: DMA address for hardware access
 * @size: Size of the buffer in bytes
 * @direction: DMA direction (DMA_TO_DEVICE, DMA_FROM_DEVICE)
 * @sg_table: Scatter-gather table for non-contiguous buffers
 * @pages: Array of pages for user-mapped buffers
 * @nr_pages: Number of pages in @pages array
 *
 * This structure describes a DMA buffer that can be accessed by both
 * the CPU and the device. The buffer can be allocated using either
 * dma_alloc_coherent() for coherent buffers or dma_alloc_attrs()
 * for streaming buffers.
 */
struct driver_buffer {
	void *cpu_addr;
	dma_addr_t dma_addr;
	size_t size;
	enum dma_data_direction direction;
	struct sg_table *sg_table;
	struct page **pages;
	unsigned int nr_pages;
};

/**
 * driver_submit_transfer() - Submit a DMA transfer
 * @dev: Driver device structure
 * @buf: DMA buffer to transfer
 * @offset: Offset within the buffer
 * @len: Length of data to transfer
 * @callback: Completion callback (may be NULL)
 * @context: Context passed to callback
 *
 * Submit a DMA transfer request to the hardware. The transfer is
 * queued and will be processed asynchronously. If a callback is
 * provided, it will be called from interrupt context when the
 * transfer completes.
 *
 * The caller must ensure that the buffer remains valid until the
 * transfer completes. Use driver_cancel_transfer() to cancel a
 * pending transfer.
 *
 * Context: Can be called from any context. Takes dev->hw_lock.
 * Return: 0 on success, negative error code on failure.
 *         -EINVAL if parameters are invalid
 *         -EBUSY if hardware queue is full
 *         -EIO if hardware is not responding
 */
int driver_submit_transfer(struct driver_device *dev,
			   struct driver_buffer *buf,
			   size_t offset, size_t len,
			   void (*callback)(void *context, int status),
			   void *context)
{
	/* Implementation */
}

/**
 * DOC: Driver Architecture
 *
 * This driver implements a character device interface for the XYZ
 * hardware accelerator. The driver architecture consists of:
 *
 * 1. **Device Discovery**: Platform device or PCI enumeration
 * 2. **Memory Management**: DMA buffer allocation and mapping
 * 3. **Command Submission**: Asynchronous command queue
 * 4. **Interrupt Handling**: Top/bottom half processing
 * 5. **Power Management**: Runtime PM and system suspend/resume
 *
 * Locking Model
 * =============
 *
 * The driver uses the following locks:
 * - dev->lock (mutex): Protects device state and configuration
 * - dev->hw_lock (spinlock): Protects hardware register access
 * - dev->list_lock (spinlock): Protects command queue
 *
 * Lock ordering: lock -> list_lock -> hw_lock
 *
 * Memory Management
 * =================
 *
 * The driver supports both coherent and streaming DMA buffers.
 * Coherent buffers are used for command rings and status blocks.
 * Streaming buffers are used for data transfers.
 */
```

**Required Documentation**:

- All exported functions (EXPORT_SYMBOL)
- All data structures in headers
- All sysfs attributes
- Module parameters
- Complex internal functions
- Locking requirements

## Test-Driven Development (TDD) for Kernel Code

### Core TDD Requirements

**MANDATORY TDD WORKFLOW:**

1. **RED PHASE**: Write failing tests FIRST (kunit/kselftest)
2. **GREEN PHASE**: Write minimal kernel code to pass tests
3. **REFACTOR PHASE**: Improve code maintaining test success

**Testing Requirements:**

- Write KUnit tests for all internal functions
- Write kselftest for user-space interfaces
- Test all error paths and edge cases
- Mock hardware interactions
- Validate locking correctness

### Testing Frameworks

- **Unit Testing**: KUnit (in-kernel unit testing)
- **Integration Testing**: kselftest framework
- **Stress Testing**: Custom stress test modules
- **Coverage**: gcov kernel support
- **Static Analysis**: sparse, smatch, coccinelle

### KUnit Test Structure

```c
// drivers/mydriver/tests/driver_test.c
#include <kunit/test.h>
#include <kunit/mock.h>
#include "../driver_core.h"

/* Test fixture for driver tests */
struct driver_test_context {
	struct driver_device *dev;
	struct mock_hw *mock_hw;
};

static int driver_test_init(struct kunit *test)
{
	struct driver_test_context *ctx;
	
	ctx = kunit_kzalloc(test, sizeof(*ctx), GFP_KERNEL);
	KUNIT_ASSERT_NOT_ERR_OR_NULL(test, ctx);
	
	ctx->mock_hw = create_mock_hardware(test);
	KUNIT_ASSERT_NOT_ERR_OR_NULL(test, ctx->mock_hw);
	
	ctx->dev = driver_device_create(ctx->mock_hw);
	KUNIT_ASSERT_NOT_ERR_OR_NULL(test, ctx->dev);
	
	test->priv = ctx;
	return 0;
}

static void driver_test_exit(struct kunit *test)
{
	struct driver_test_context *ctx = test->priv;
	
	driver_device_destroy(ctx->dev);
	destroy_mock_hardware(ctx->mock_hw);
}

/* Test cases */
static void driver_test_buffer_alloc_success(struct kunit *test)
{
	struct driver_test_context *ctx = test->priv;
	struct driver_buffer *buf;
	
	/* Test normal allocation */
	buf = driver_alloc_buffer(ctx->dev, PAGE_SIZE);
	KUNIT_ASSERT_NOT_ERR_OR_NULL(test, buf);
	KUNIT_EXPECT_EQ(test, buf->size, PAGE_SIZE);
	KUNIT_EXPECT_NOT_NULL(test, buf->cpu_addr);
	KUNIT_EXPECT_NE(test, buf->dma_addr, 0);
	
	driver_free_buffer(ctx->dev, buf);
}

static void driver_test_buffer_alloc_too_large(struct kunit *test)
{
	struct driver_test_context *ctx = test->priv;
	struct driver_buffer *buf;
	
	/* Test allocation beyond limit */
	buf = driver_alloc_buffer(ctx->dev, MAX_BUFFER_SIZE + 1);
	KUNIT_EXPECT_PTR_EQ(test, buf, ERR_PTR(-EINVAL));
}

static void driver_test_concurrent_access(struct kunit *test)
{
	struct driver_test_context *ctx = test->priv;
	struct task_struct *threads[4];
	struct completion done;
	atomic_t counter;
	int i;
	
	init_completion(&done);
	atomic_set(&counter, 0);
	
	/* Spawn concurrent threads */
	for (i = 0; i < ARRAY_SIZE(threads); i++) {
		threads[i] = kthread_run(concurrent_test_thread,
					 ctx->dev, "test-%d", i);
		KUNIT_ASSERT_NOT_ERR_OR_NULL(test, threads[i]);
	}
	
	/* Wait for completion */
	for (i = 0; i < ARRAY_SIZE(threads); i++) {
		kthread_stop(threads[i]);
	}
	
	/* Verify results */
	KUNIT_EXPECT_EQ(test, atomic_read(&counter), ARRAY_SIZE(threads));
}

/* Parameterized tests */
static const struct driver_dma_test_case {
	const char *name;
	size_t size;
	enum dma_data_direction dir;
	int expected_result;
} dma_test_cases[] = {
	{
		.name = "small_to_device",
		.size = 512,
		.dir = DMA_TO_DEVICE,
		.expected_result = 0,
	},
	{
		.name = "large_from_device",
		.size = 64 * 1024,
		.dir = DMA_FROM_DEVICE,
		.expected_result = 0,
	},
	{
		.name = "invalid_direction",
		.size = PAGE_SIZE,
		.dir = DMA_NONE,
		.expected_result = -EINVAL,
	},
};

static void driver_test_dma_transfer(struct kunit *test)
{
	const struct driver_dma_test_case *params = test->param_value;
	struct driver_test_context *ctx = test->priv;
	struct driver_buffer *buf;
	int ret;
	
	buf = driver_alloc_buffer(ctx->dev, params->size);
	KUNIT_ASSERT_NOT_ERR_OR_NULL(test, buf);
	
	ret = driver_setup_dma(ctx->dev, buf, params->dir);
	KUNIT_EXPECT_EQ(test, ret, params->expected_result);
	
	if (ret == 0) {
		ret = driver_start_dma(ctx->dev, buf);
		KUNIT_EXPECT_EQ(test, ret, 0);
		
		/* Verify mock hardware was programmed correctly */
		KUNIT_EXPECT_EQ(test, ctx->mock_hw->dma_addr, buf->dma_addr);
		KUNIT_EXPECT_EQ(test, ctx->mock_hw->dma_size, buf->size);
		KUNIT_EXPECT_EQ(test, ctx->mock_hw->dma_dir, params->dir);
	}
	
	driver_free_buffer(ctx->dev, buf);
}

/* Test suite definition */
static struct kunit_case driver_test_cases[] = {
	KUNIT_CASE(driver_test_buffer_alloc_success),
	KUNIT_CASE(driver_test_buffer_alloc_too_large),
	KUNIT_CASE(driver_test_concurrent_access),
	KUNIT_CASE_PARAM(driver_test_dma_transfer, dma_test_gen_params),
	{}
};

static struct kunit_suite driver_test_suite = {
	.name = "driver-tests",
	.init = driver_test_init,
	.exit = driver_test_exit,
	.test_cases = driver_test_cases,
};

kunit_test_suites(&driver_test_suite);

MODULE_LICENSE("GPL");
MODULE_DESCRIPTION("Driver KUnit tests");
```

### Kselftest Implementation

```c
// tools/testing/selftests/drivers/mydriver/test_ioctl.c
#include <stdio.h>
#include <stdlib.h>
#include <fcntl.h>
#include <unistd.h>
#include <sys/ioctl.h>
#include <errno.h>
#include <string.h>
#include <pthread.h>
#include "../../../../../include/uapi/linux/mydriver.h"
#include "../../kselftest.h"

#define DEVICE_PATH "/dev/mydriver0"

static int device_fd;

static void test_open_close(void)
{
	int fd;
	
	fd = open(DEVICE_PATH, O_RDWR);
	if (fd < 0) {
		ksft_test_result_fail("Failed to open device: %s\n",
				      strerror(errno));
		return;
	}
	
	if (close(fd) != 0) {
		ksft_test_result_fail("Failed to close device: %s\n",
				      strerror(errno));
		return;
	}
	
	ksft_test_result_pass("Open/close device\n");
}

static void test_invalid_ioctl(void)
{
	int ret;
	
	ret = ioctl(device_fd, 0xDEADBEEF, NULL);
	if (ret != -1 || errno != ENOTTY) {
		ksft_test_result_fail("Invalid ioctl returned %d, errno %d\n",
				      ret, errno);
		return;
	}
	
	ksft_test_result_pass("Invalid ioctl rejected\n");
}

static void test_buffer_operations(void)
{
	struct mydriver_buffer_info buf_info = {
		.size = 4096,
		.flags = MYDRIVER_BUF_CACHED,
	};
	void *mapped;
	int ret;
	
	/* Allocate buffer */
	ret = ioctl(device_fd, MYDRIVER_IOCTL_ALLOC_BUFFER, &buf_info);
	if (ret < 0) {
		ksft_test_result_fail("Buffer allocation failed: %s\n",
				      strerror(errno));
		return;
	}
	
	/* Map buffer */
	mapped = mmap(NULL, buf_info.size, PROT_READ | PROT_WRITE,
		      MAP_SHARED, device_fd, buf_info.offset);
	if (mapped == MAP_FAILED) {
		ksft_test_result_fail("Buffer mapping failed: %s\n",
				      strerror(errno));
		goto free_buffer;
	}
	
	/* Test buffer access */
	memset(mapped, 0xAA, buf_info.size);
	if (*(unsigned char *)mapped != 0xAA) {
		ksft_test_result_fail("Buffer write/read mismatch\n");
		goto unmap_buffer;
	}
	
	ksft_test_result_pass("Buffer operations\n");
	
unmap_buffer:
	munmap(mapped, buf_info.size);
free_buffer:
	ioctl(device_fd, MYDRIVER_IOCTL_FREE_BUFFER, &buf_info.handle);
}

static void *stress_thread(void *arg)
{
	int thread_id = (int)(intptr_t)arg;
	struct mydriver_command cmd = {
		.opcode = MYDRIVER_OP_PROCESS,
		.size = 1024,
	};
	int i, ret;
	
	for (i = 0; i < 1000; i++) {
		cmd.data = (void *)(intptr_t)(thread_id * 1000 + i);
		ret = ioctl(device_fd, MYDRIVER_IOCTL_SUBMIT_CMD, &cmd);
		if (ret < 0) {
			ksft_print_msg("Thread %d iter %d failed: %s\n",
				       thread_id, i, strerror(errno));
			return (void *)-1;
		}
	}
	
	return NULL;
}

static void test_concurrent_access(void)
{
	pthread_t threads[8];
	void *thread_ret;
	int i, ret;
	int failed = 0;
	
	/* Create threads */
	for (i = 0; i < 8; i++) {
		ret = pthread_create(&threads[i], NULL, stress_thread,
				     (void *)(intptr_t)i);
		if (ret != 0) {
			ksft_test_result_fail("Failed to create thread %d\n", i);
			return;
		}
	}
	
	/* Wait for threads */
	for (i = 0; i < 8; i++) {
		pthread_join(threads[i], &thread_ret);
		if (thread_ret != NULL)
			failed++;
	}
	
	if (failed > 0)
		ksft_test_result_fail("Concurrent access: %d threads failed\n",
				      failed);
	else
		ksft_test_result_pass("Concurrent access\n");
}

int main(void)
{
	ksft_print_header();
	ksft_set_plan(4);
	
	/* Open device for tests */
	device_fd = open(DEVICE_PATH, O_RDWR);
	if (device_fd < 0) {
		ksft_exit_skip("Cannot open device %s: %s\n",
			       DEVICE_PATH, strerror(errno));
	}
	
	/* Run tests */
	test_open_close();
	test_invalid_ioctl();
	test_buffer_operations();
	test_concurrent_access();
	
	close(device_fd);
	
	ksft_finished();
	return 0;
}
```

### Mock Hardware Implementation

```c
// drivers/mydriver/tests/mock_hardware.h
#ifndef _MOCK_HARDWARE_H
#define _MOCK_HARDWARE_H

#include <linux/types.h>
#include <kunit/test.h>

struct mock_hw {
	/* Mocked registers */
	u32 status_reg;
	u32 control_reg;
	u32 dma_addr_reg;
	u32 dma_size_reg;
	
	/* State tracking */
	bool irq_enabled;
	int irq_count;
	
	/* DMA tracking */
	dma_addr_t dma_addr;
	size_t dma_size;
	enum dma_data_direction dma_dir;
	
	/* Error injection */
	bool fail_next_dma;
	int error_code;
	
	/* KUnit test context */
	struct kunit *test;
};

/* Mock hardware operations */
static inline u32 mock_hw_read32(struct mock_hw *hw, unsigned int reg)
{
	switch (reg) {
	case DRIVER_STATUS_REG:
		return hw->status_reg;
	case DRIVER_CONTROL_REG:
		return hw->control_reg;
	case DRIVER_DMA_ADDR_REG:
		return hw->dma_addr_reg;
	case DRIVER_DMA_SIZE_REG:
		return hw->dma_size_reg;
	default:
		KUNIT_FAIL(hw->test, "Invalid register read: 0x%x", reg);
		return 0xDEADBEEF;
	}
}

static inline void mock_hw_write32(struct mock_hw *hw, u32 val, unsigned int reg)
{
	switch (reg) {
	case DRIVER_CONTROL_REG:
		hw->control_reg = val;
		if (val & DRIVER_CTRL_START_DMA) {
			if (hw->fail_next_dma) {
				hw->status_reg |= DRIVER_STATUS_ERROR;
				hw->fail_next_dma = false;
			} else {
				hw->status_reg |= DRIVER_STATUS_DMA_DONE;
			}
			if (hw->irq_enabled) {
				hw->irq_count++;
				/* Trigger mock interrupt */
			}
		}
		break;
	case DRIVER_DMA_ADDR_REG:
		hw->dma_addr_reg = val;
		hw->dma_addr = val;
		break;
	case DRIVER_DMA_SIZE_REG:
		hw->dma_size_reg = val;
		hw->dma_size = val;
		break;
	default:
		KUNIT_FAIL(hw->test, "Invalid register write: 0x%x = 0x%x",
			   reg, val);
	}
}

static struct mock_hw *create_mock_hardware(struct kunit *test)
{
	struct mock_hw *hw;
	
	hw = kunit_kzalloc(test, sizeof(*hw), GFP_KERNEL);
	KUNIT_ASSERT_NOT_NULL(test, hw);
	
	hw->test = test;
	hw->status_reg = DRIVER_STATUS_READY;
	
	return hw;
}

static void destroy_mock_hardware(struct mock_hw *hw)
{
	/* Cleanup handled by KUnit allocator */
}

/* Macro to replace real hardware access in tests */
#define readl(addr) mock_hw_read32(dev->mock_hw, (unsigned long)(addr))
#define writel(val, addr) mock_hw_write32(dev->mock_hw, val, (unsigned long)(addr))

#endif /* _MOCK_HARDWARE_H */
```

## Length Restrictions

### Files

- **Maximum Lines**: 1000 (kernel modules can be larger)
- **Enforcement**: Strict
- **Exceptions**:
  - Auto-generated files (device tables, firmware)
  - Complex drivers with extensive hardware support
  - Headers with many inline functions

**Refactoring Strategies**:

- Split driver into core + function-specific files
- Separate ioctl handlers into dedicated file
- Extract sysfs attributes to separate file
- Move debug/trace code to separate file
- Use separate files for each major subsystem

### Functions

- **Maximum Lines**: 100 (kernel allows slightly larger)
- **Enforcement**: Strict
- **Exceptions**:
  - Complex initialization sequences
  - State machines with extensive documentation
  - Hardware workaround implementations

**Refactoring Strategies**:

- Extract helper functions for repeated code
- Use function pointers for variant behavior
- Split complex algorithms into steps
- Move error handling to separate functions
- Use inline functions for small operations

### Structures

- **Maximum Members**: 20
- **Cache Line Alignment**: Consider for performance-critical structs
- **Guidance**: Group related fields, minimize padding

## Linting Configuration

### Primary Tools

- **Static Analysis**: sparse, smatch
- **Style Checker**: checkpatch.pl
- **Code Formatter**: clang-format
- **Additional Tools**:
  - Coccinelle (semantic patches)
  - cppcheck
  - flawfinder (security)
  - scripts/kernel-doc (documentation)

### Sparse Configuration

**Running sparse**:

```bash
# Enable sparse checking
make C=1        # Check only re-compiled files
make C=2        # Force check of all files

# With specific checks
make C=2 CF="-Wbitwise -Wno-decl"
```

### Checkpatch Configuration

**Configuration** (`.checkpatch.conf`):

```
# Checkpatch configuration for kernel modules
--strict
--no-tree
--show-types
--max-line-length=80
--ignore CONST_STRUCT
--ignore PREFER_KERNEL_TYPES
```

**Pre-commit script**:

```bash
#!/bin/bash
# .git/hooks/pre-commit

# Run checkpatch on staged files
for file in $(git diff --cached --name-only | grep -E '\.[ch]$'); do
    if ! scripts/checkpatch.pl --strict --file "$file"; then
        echo "Checkpatch failed on $file"
        exit 1
    fi
done

# Run sparse
make C=2 2>&1 | grep -v "^  CHECK" | grep -v "^  CC"
if [ ${PIPESTATUS[0]} -ne 0 ]; then
    echo "Sparse check failed"
    exit 1
fi
```

### Clang-Format Configuration

**Configuration** (`.clang-format`):

```yaml
# Linux kernel style
BasedOnStyle: LLVM
IndentWidth: 8
UseTab: Always
BreakBeforeBraces: Linux
AllowShortIfStatementsOnASingleLine: false
IndentCaseLabels: false
ColumnLimit: 80
AlignAfterOpenBracket: Align
AlignConsecutiveMacros: true
AlignConsecutiveAssignments: false
AlignEscapedNewlines: Left
AlignOperands: true
AlignTrailingComments: true
AllowShortBlocksOnASingleLine: false
AllowShortFunctionsOnASingleLine: None
AlwaysBreakAfterReturnType: None
AlwaysBreakBeforeMultilineStrings: false
BinPackArguments: true
BinPackParameters: true
BreakBeforeBinaryOperators: None
BreakStringLiterals: true
CommentPragmas: '^ IWYU pragma:'
ContinuationIndentWidth: 8
Cpp11BracedListStyle: false
DerivePointerAlignment: false
DisableFormat: false
ForEachMacros:
  - 'list_for_each'
  - 'list_for_each_entry'
  - 'list_for_each_entry_safe'
  - 'hlist_for_each'
  - 'hlist_for_each_entry'
IncludeBlocks: Preserve
IndentGotoLabels: false
IndentPPDirectives: None
IndentWrappedFunctionNames: false
KeepEmptyLinesAtTheStartOfBlocks: false
MacroBlockBegin: ''
MacroBlockEnd: ''
MaxEmptyLinesToKeep: 1
PenaltyBreakAssignment: 10
PenaltyBreakBeforeFirstCallParameter: 19
PenaltyBreakComment: 300
PenaltyBreakFirstLessLess: 120
PenaltyBreakString: 1000
PenaltyExcessCharacter: 1000000
PenaltyReturnTypeOnItsOwnLine: 60
PointerAlignment: Right
ReflowComments: true
SortIncludes: false
SpaceAfterCStyleCast: false
SpaceBeforeAssignmentOperators: true
SpaceBeforeParens: ControlStatements
SpaceInEmptyParentheses: false
SpacesBeforeTrailingComments: 1
SpacesInContainerLiterals: false
SpacesInCStyleCastParentheses: false
SpacesInParentheses: false
SpacesInSquareBrackets: false
Standard: Cpp11
```

### Coccinelle Scripts

**Common semantic patches** (`scripts/coccinelle/`):

```cocci
// api/alloc/kzalloc-simple.cocci
@@
expression x, E1, E2;
@@
- x = kmalloc(E1, E2);
+ x = kzalloc(E1, E2);
  ...
- memset(x, 0, E1);

// api/err_ptr.cocci
@@
expression x;
@@
- if (IS_ERR(x) || !x)
+ if (IS_ERR_OR_NULL(x))

// free/kfree.cocci
@@
expression x;
@@
- if (x != NULL)
    kfree(x);
+ kfree(x);
```

### Makefile Integration

**Makefile**:

```makefile
# Kernel module Makefile with testing support

# Module name
MODULE_NAME := mydriver

# Source files
$(MODULE_NAME)-y := driver_core.o driver_ioctl.o driver_sysfs.o

# Test files (KUnit)
$(MODULE_NAME)-$(CONFIG_KUNIT) += tests/driver_test.o tests/mock_hardware.o

# Object files
obj-m += $(MODULE_NAME).o

# Kernel directory
KERNELDIR ?= /lib/modules/$(shell uname -r)/build
PWD := $(shell pwd)

# Compilation flags
ccflags-y := -Wall -Werror
ccflags-$(CONFIG_DEBUG) += -g -DDEBUG

# Sparse flags
CF := -Wbitwise -Wno-return-void -Wno-unknown-attribute

.PHONY: all clean sparse checkpatch test install

all:
	$(MAKE) -C $(KERNELDIR) M=$(PWD) modules

clean:
	$(MAKE) -C $(KERNELDIR) M=$(PWD) clean
	rm -f *.o.ur-safe

# Static analysis
sparse:
	$(MAKE) -C $(KERNELDIR) M=$(PWD) C=2 CF="$(CF)" modules

# Style checking
checkpatch:
	@for file in *.c *.h; do \
		echo "Checking $$file..."; \
		scripts/checkpatch.pl --strict --file $$file || exit 1; \
	done

# Run KUnit tests
test:
	@echo "Building with KUnit tests..."
	$(MAKE) -C $(KERNELDIR) M=$(PWD) CONFIG_KUNIT=y modules
	@echo "Running KUnit tests..."
	./tools/testing/kunit/kunit.py run --kunitconfig=$(PWD)/.kunitconfig

# Run kselftest
test-user:
	$(MAKE) -C tools/testing/selftests/drivers/$(MODULE_NAME) run_tests

# Code formatting
format:
	@find . -name "*.c" -o -name "*.h" | xargs clang-format -i

# Check formatting
format-check:
	@find . -name "*.c" -o -name "*.h" | xargs clang-format --dry-run --Werror

# Generate documentation
docs:
	@for file in *.c; do \
		scripts/kernel-doc -rst $$file > docs/$$file.rst; \
	done

# Full quality check
quality: checkpatch sparse format-check test

# Install module
install: all
	$(MAKE) -C $(KERNELDIR) M=$(PWD) modules_install
	depmod -a

# Development workflow
dev: format sparse checkpatch
	@echo "Development checks passed!"

# CI workflow
ci: format-check checkpatch sparse test test-user
	@echo "CI checks passed!"
```

## Quality Gates

### Pre-commit Requirements (Enforced by TDD)

- Code passes `checkpatch.pl --strict`
- No sparse warnings
- clang-format compliant
- All KUnit tests pass
- Kselftest suite passes
- Documentation complete (kernel-doc)
- No memory leaks (kmemleak clean)
- Lockdep warnings resolved

### Continuous Integration (GitHub Actions)

```yaml
# .github/workflows/kernel-ci.yml
name: Kernel Module CI

on:
  push:
    branches: [main, develop]
  pull_request:
    branches: [main]

jobs:
  build:
    runs-on: ubuntu-25.04
    strategy:
      matrix:
        kernel: ['6.8', '6.9', '6.10']
        arch: ['x86_64', 'arm64']
    
    steps:
    - uses: actions/checkout@v4
    
    - name: Install dependencies
      run: |
        sudo apt-get update
        sudo apt-get install -y \
          linux-headers-generic \
          sparse \
          clang-format \
          coccinelle \
          cppcheck
    
    - name: Install kernel ${{ matrix.kernel }}
      run: |
        wget https://kernel.org/pub/linux/kernel/v6.x/linux-${{ matrix.kernel }}.tar.xz
        tar -xf linux-${{ matrix.kernel }}.tar.xz
        cd linux-${{ matrix.kernel }}
        make defconfig
        make modules_prepare
    
    - name: Check code format
      run: make format-check
    
    - name: Run checkpatch
      run: make checkpatch
    
    - name: Run sparse
      run: make sparse KERNELDIR=../linux-${{ matrix.kernel }}
    
    - name: Build module
      run: make all KERNELDIR=../linux-${{ matrix.kernel }}
    
    - name: Run KUnit tests
      run: |
        cd linux-${{ matrix.kernel }}
        ./tools/testing/kunit/kunit.py run \
          --kunitconfig=../kunit.config \
          --arch=${{ matrix.arch }}
    
    - name: Run Coccinelle
      run: |
        for script in scripts/coccinelle/api/*.cocci; do
          spatch --sp-file $script --dir . --in-place
        done
        git diff --exit-code
    
    - name: Security scan
      run: |
        cppcheck --enable=all --error-exitcode=1 .
        flawfinder --error-level=1 .

  test:
    runs-on: ubuntu-25.04
    needs: build
    
    steps:
    - uses: actions/checkout@v4
    
    - name: Setup test VM
      run: |
        # Setup QEMU VM for testing
        wget https://cloud-images.ubuntu.com/minimal/releases/25.04/release/ubuntu-25.04-minimal-cloudimg-amd64.img
        qemu-img resize ubuntu-25.04-minimal-cloudimg-amd64.img 10G
    
    - name: Run integration tests
      run: |
        # Boot VM and run tests
        ./scripts/run-vm-tests.sh
    
    - name: Stress testing
      run: |
        # Run stress tests in VM
        ./scripts/stress-test.sh --duration 300
```

## Project Structure

### Recommended Directory Layout

```
mydriver/
├── Makefile                # Main build file
├── Kconfig                 # Kernel configuration
├── .clang-format          # Formatting config
├── .checkpatch.conf       # Checkpatch config
├── .kunitconfig           # KUnit test config
├── driver_core.c          # Core driver functionality
├── driver_core.h          # Internal headers
├── driver_hw.c            # Hardware abstraction
├── driver_hw.h            # Hardware definitions
├── driver_ioctl.c         # ioctl interface
├── driver_ioctl.h         # ioctl definitions
├── driver_sysfs.c         # sysfs attributes
├── driver_sysfs.h         # sysfs helpers
├── driver_debug.c         # Debug/trace support
├── driver_debug.h         # Debug macros
├── tests/                 # KUnit tests
│   ├── driver_test.c      # Main test suite
│   ├── mock_hardware.h    # Hardware mocking
│   └── test_helpers.h     # Test utilities
├── tools/                 # User-space tools
│   ├── testing/           # Testing tools
│   │   └── selftests/     # Kselftest suite
│   │       └── drivers/
│   │           └── mydriver/
│   │               ├── Makefile
│   │               ├── test_basic.c
│   │               ├── test_ioctl.c
│   │               └── test_stress.c
│   └── utils/             # Driver utilities
│       ├── mydriver-ctl   # Control utility
│       └── mydriver-mon   # Monitoring tool
├── scripts/               # Helper scripts
│   ├── run-tests.sh       # Test runner
│   ├── run-vm-tests.sh    # VM test runner
│   └── stress-test.sh     # Stress testing
├── docs/                  # Documentation
│   ├── design.md          # Architecture doc
│   ├── api.md             # API reference
│   └── testing.md         # Testing guide
└── examples/              # Example code
    ├── basic_usage.c      # Basic example
    └── advanced_dma.c     # DMA example
```

## Debugging and Tracing

### Debug Infrastructure

```c
/* driver_debug.h - Debug and trace infrastructure */
#ifndef _DRIVER_DEBUG_H
#define _DRIVER_DEBUG_H

#include <linux/printk.h>
#include <linux/dynamic_debug.h>

/* Debug levels */
#define DRIVER_DBG_INIT		0x0001
#define DRIVER_DBG_PROBE	0x0002
#define DRIVER_DBG_IO		0x0004
#define DRIVER_DBG_IRQ		0x0008
#define DRIVER_DBG_DMA		0x0010
#define DRIVER_DBG_REG		0x0020

/* Module parameter for debug mask */
extern unsigned int debug_mask;

/* Debug macros */
#define driver_dbg(mask, fmt, ...) \
do { \
	if (unlikely(debug_mask & (mask))) \
		pr_debug("[%s:%d] " fmt, __func__, __LINE__, ##__VA_ARGS__); \
} while (0)

#define driver_enter() driver_dbg(DRIVER_DBG_IO, "enter\n")
#define driver_exit() driver_dbg(DRIVER_DBG_IO, "exit\n")

/* Trace points */
#ifdef CONFIG_DRIVER_TRACE
#include <trace/events/driver.h>
#else
#define trace_driver_command(cmd) do {} while (0)
#define trace_driver_complete(cmd, status) do {} while (0)
#endif

/* Register access logging */
#ifdef CONFIG_DRIVER_REG_DEBUG
#define driver_read32(addr) ({ \
	u32 __val = readl(addr); \
	driver_dbg(DRIVER_DBG_REG, "read32 [%p] = 0x%08x\n", addr, __val); \
	__val; \
})

#define driver_write32(val, addr) do { \
	driver_dbg(DRIVER_DBG_REG, "write32 [%p] = 0x%08x\n", addr, val); \
	writel(val, addr); \
} while (0)
#else
#define driver_read32(addr) readl(addr)
#define driver_write32(val, addr) writel(val, addr)
#endif

/* Hexdump for debugging */
static inline void driver_hexdump(const char *prefix, const void *buf, size_t len)
{
	if (debug_mask)
		print_hex_dump_debug(prefix, DUMP_PREFIX_OFFSET, 16, 1,
				     buf, len, true);
}

#endif /* _DRIVER_DEBUG_H */
```

### Ftrace Integration

```c
/* trace/events/driver.h - Ftrace events */
#undef TRACE_SYSTEM
#define TRACE_SYSTEM driver

#if !defined(_TRACE_DRIVER_H) || defined(TRACE_HEADER_MULTI_READ)
#define _TRACE_DRIVER_H

#include <linux/tracepoint.h>

TRACE_EVENT(driver_command,
	TP_PROTO(struct driver_command *cmd),
	TP_ARGS(cmd),
	
	TP_STRUCT__entry(
		__field(u32, opcode)
		__field(u32, size)
		__field(void *, data)
		__field(u64, timestamp)
	),
	
	TP_fast_assign(
		__entry->opcode = cmd->opcode;
		__entry->size = cmd->size;
		__entry->data = cmd->data;
		__entry->timestamp = ktime_get_ns();
	),
	
	TP_printk("opcode=%u size=%u data=%p time=%llu",
		  __entry->opcode, __entry->size, __entry->data,
		  __entry->timestamp)
);

TRACE_EVENT(driver_irq,
	TP_PROTO(int irq, u32 status),
	TP_ARGS(irq, status),
	
	TP_STRUCT__entry(
		__field(int, irq)
		__field(u32, status)
	),
	
	TP_fast_assign(
		__entry->irq = irq;
		__entry->status = status;
	),
	
	TP_printk("irq=%d status=0x%08x", __entry->irq, __entry->status)
);

#endif /* _TRACE_DRIVER_H */

/* This part must be outside protection */
#undef TRACE_INCLUDE_PATH
#define TRACE_INCLUDE_PATH ../../drivers/mydriver
#define TRACE_INCLUDE_FILE driver_trace
#include <trace/define_trace.h>
```

## Security Hardening

### Input Validation

```c
/* Secure input validation for kernel code */

/* Validate user pointer and size */
static int validate_user_buffer(const void __user *buf, size_t size)
{
	/* Check for NULL pointer */
	if (!buf)
		return -EINVAL;
	
	/* Check for overflow */
	if (size > MAX_USER_BUFFER_SIZE)
		return -E2BIG;
	
	/* Check user space access */
	if (!access_ok(buf, size))
		return -EFAULT;
	
	return 0;
}

/* Safe string copy from user */
static int copy_string_from_user(char *dst, const char __user *src, size_t max_len)
{
	long ret;
	
	ret = strncpy_from_user(dst, src, max_len);
	if (ret < 0)
		return ret;
	
	if (ret == max_len) {
		dst[max_len - 1] = '\0';
		return -E2BIG;
	}
	
	return 0;
}

/* Integer overflow checks */
static inline bool check_mul_overflow(size_t a, size_t b, size_t *result)
{
	return __builtin_mul_overflow(a, b, result);
}

/* Bounds checking */
static inline int check_bounds(unsigned int val, unsigned int min, unsigned int max)
{
	if (val < min || val > max)
		return -ERANGE;
	return 0;
}
```

### Permission Checks

```c
/* Security and permission validation */

static int driver_check_permissions(struct file *file, unsigned int cmd)
{
	/* Check basic permissions */
	if (!capable(CAP_SYS_ADMIN)) {
		/* Non-privileged commands */
		switch (cmd) {
		case DRIVER_IOCTL_GET_INFO:
		case DRIVER_IOCTL_GET_STATUS:
			break;
		default:
			return -EPERM;
		}
	}
	
	/* Check SELinux permissions if enabled */
#ifdef CONFIG_SECURITY_SELINUX
	return security_file_ioctl(file, cmd, 0);
#else
	return 0;
#endif
}

/* Restrict device access */
static char *driver_devnode(const struct device *dev, umode_t *mode)
{
	if (mode)
		*mode = 0600; /* rw------- */
	return NULL;
}
```

---

_This document serves as the comprehensive coding standard for C-based kernel development on Ubuntu 25.04 with mandatory Test-Driven Development. All code must follow the TDD workflow: write tests FIRST using KUnit/kselftest, then implementation, maintaining the kernel coding standards and quality gates defined herein._
