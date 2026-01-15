import re
import argparse

from elftools.elf.elffile import ELFFile
from elftools.elf.sections import SymbolTableSection



class elf2header(object):
    def __init__(self, verbose=False):
        self.binaries = []
        self.mem = {}
        self.verbose = verbose
        self.areas = []
        self.start_addr = 0
        self.header = None

        if self.verbose:
            print("elf2header")
            
            
    def add_header(self, header):
        print("  Added header: %s" % header)
        self.header = header


    def get_entry(self):
        with open(self.binaries[0], "rb") as file:
            elffile = ELFFile(file)
            return elffile.header["e_entry"]


    def add_binary(self, binary):
        print("  Added binary: %s" % binary)
        self.binaries.append(binary)


    def add_area(self, start, size):
        print("  Added target area: [0x%x -> 0x%x]" % (start, start + size))
        self.areas.append([start, start + size])


    def __add_mem_word(self, base, size, data, width):
        aligned_base = base & ~(width - 1)

        shift = base - aligned_base
        iter_size = width - shift
        if iter_size > size:
            iter_size = size

        value = self.mem.get(str(aligned_base))
        if value is None:
            value = 0

        value &= ~(((1 << width) - 1) << (shift * 8))
        value |= int.from_bytes(data[0:iter_size], byteorder="little") << (shift * 8)

        self.mem[str(aligned_base)] = value

        return iter_size


    def _add_mem(self, base, size, data, width):
        while size > 0:
            iter_size = self.__add_mem_word(base, size, data, width)

            size -= iter_size
            base += iter_size
            data = data[iter_size:]


    def _generate_dma_header(self, filename, width):
        print("  Generating DMA header to file: " + filename)
        
        header_str = open(self.header, "r").read()
        
        node_name_re = r"@name_begin@\s+(.+)\s+@name_end@"
        node_name = re.search(node_name_re, header_str).group(1)
            
        # Group contiguous memory regions for efficient DMA transfers
        regions = []
        current_region = None
        current_data = []
        
        for key in sorted(self.mem.keys(), key=lambda x: int(x)):
            addr = int(key)
            value = self.mem.get(key)
            
            # Convert the value to bytes
            #value_bytes = value.to_bytes(width, byteorder="little")
            
            if current_region is None:
                current_region = [addr, addr + width]
                current_data = [value]
            elif addr == current_region[1]:
                # Extend current region
                current_region[1] = addr + width
                current_data.append(value)
            else:
                # Start a new region after saving the current one
                regions.append((current_region[0], current_region[1], current_data))
                current_region = [addr, addr + width]
                current_data = [value]
        
        # Add the last region if there is one
        if current_region:
            regions.append((current_region[0], current_region[1], current_data))
            
        # Set Boot Address
        header_str = re.sub(r"(boot_addr = )(.*;)", lambda m: f"{m.group(1)}(void*)0x{self.start_addr:08X};", header_str)
        header_str = re.sub(r"(args_addr = )(.*;)", lambda m: f"{m.group(1)}(void*)offload_args;", header_str)

        # Add binary data sections
        regions_str = ""
        for i, (start, end, data) in enumerate(regions):
            size = len(data)
            regions_str += f"static const uint32_t {node_name}_binary_data_{i}[0x{size:08X}] __attribute__((section(\".offload\"))) = {{\n    "
            
            # Write the binary data as a hex array
            for j, value in enumerate(data):
                if j > 0 and j % 4 == 0:
                    regions_str += "\n    "
                regions_str += f"0x{value:08X}, "
            
            regions_str += "\n};\n\n"
        pattern = re.compile(r"\/\/ @data_sections_begin@\n.*\/\/ @data_sections_end@", re.DOTALL)
        regions_str = f"// @data_sections_begin@\n\n{regions_str}// @data_sections_end@"
        header_str = re.sub(pattern, regions_str, header_str)
            
        # Add the DMA transfer array
        dma_array_str = ""
        for i, (start, end, _) in enumerate(regions):
            size = end - start
            dma_array_str += f"    {{.src = (void*){node_name}_binary_data_{i}, .dst = (void*)0x{start:08X}, "
            dma_array_str += f".size = 0x{size:08X}}},\n"
        pattern = re.compile(r"\/\/ @dma_sections_begin@\n.*\/\/ @dma_sections_end@", re.DOTALL)
        dma_array_str = f"// @dma_sections_begin@\n{dma_array_str}// @dma_sections_end@"
        header_str = re.sub(pattern, dma_array_str, header_str)
        
        with open(self.header, "w") as file:
            file.write(header_str)
            

    def _get_start_address(self):
        for binary in self.binaries:
            with open(binary, "rb") as file:
                elffile = ELFFile(file)
                for section in elffile.iter_sections():
                    if isinstance(section, SymbolTableSection):
                        for symbol in section.iter_symbols():
                            if symbol.name == "_start":
                                self.start_addr = symbol.entry["st_value"]
                                print("  Found _start address: 0x%x" % self.start_addr)
                                break


    def _parse_binary(self, width):
        self.mem = {}

        for binary in self.binaries:
            with open(binary, "rb") as file:
                elffile = ELFFile(file)

                for segment in elffile.iter_segments():
                    if segment["p_type"] == "PT_LOAD":
                        data = segment.data()
                        addr = segment["p_paddr"]
                        size = len(data)
                        load = True
                        if len(self.areas) != 0:
                            load = False
                            for area in self.areas:
                                if addr >= area[0] and addr + size <= area[1]:
                                    load = True
                                    break
                        if load:
                            print(
                                "  Handling section (base: 0x%x, size: 0x%x)"
                                % (addr, size)
                            )
                            self._add_mem(addr, size, data, width)
                            if segment["p_filesz"] < segment["p_memsz"]:
                                addr = segment["p_paddr"] + segment["p_filesz"]
                                size = segment["p_memsz"] - segment["p_filesz"]
                                print(
                                    "  Init section to 0 (base: 0x%x, size: 0x%x)"
                                    % (addr, size)
                                )
                                self._add_mem(addr, size, [0] * size, width)
                        else:
                            print(
                                "  Bypassing section (base: 0x%x, size: 0x%x)"
                                % (addr, size)
                            )


    def generate_dma_header_32(self, output_file):
        self._parse_binary(4)
        self._get_start_address()
        self._generate_dma_header(output_file, 4)



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert ELF to header file.")

    parser.add_argument(
        "--binary", dest="binary", default=None, help="Specify input binary"
    )
    parser.add_argument(
        "--output", dest="output", default=None, help="Specify output header file"
    )
   
    args = parser.parse_args()

    converter = elf2header(verbose=True)
    converter.add_binary(args.binary)
    converter.add_header(args.output)
    
    converter.generate_dma_header_32(args.output)