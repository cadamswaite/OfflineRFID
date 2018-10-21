#!/usr/bin/env python2
# -*- coding: utf-8 -*-
##################################################
# GNU Radio Python Flow Graph
# Title: Top Block
# Generated: Thu Sep 21 11:31:16 2017
##################################################

if __name__ == '__main__':
    import ctypes
    import sys
    if sys.platform.startswith('linux'):
        try:
            x11 = ctypes.cdll.LoadLibrary('libX11.so')
            x11.XInitThreads()
        except:
            print "Warning: failed to XInitThreads()"

from PyQt4 import Qt
from gnuradio import blocks
from gnuradio import eng_notation
from gnuradio import filter
from gnuradio import gr
from gnuradio.eng_option import eng_option
from gnuradio.filter import firdes
from optparse import OptionParser
import rfid
import sys
from gnuradio import qtgui


class top_block(gr.top_block, Qt.QWidget):

    def __init__(self):
        gr.top_block.__init__(self, "Top Block")
        Qt.QWidget.__init__(self)
        self.setWindowTitle("Top Block")
        qtgui.util.check_set_qss()
        try:
            self.setWindowIcon(Qt.QIcon.fromTheme('gnuradio-grc'))
        except:
            pass
        self.top_scroll_layout = Qt.QVBoxLayout()
        self.setLayout(self.top_scroll_layout)
        self.top_scroll = Qt.QScrollArea()
        self.top_scroll.setFrameStyle(Qt.QFrame.NoFrame)
        self.top_scroll_layout.addWidget(self.top_scroll)
        self.top_scroll.setWidgetResizable(True)
        self.top_widget = Qt.QWidget()
        self.top_scroll.setWidget(self.top_widget)
        self.top_layout = Qt.QVBoxLayout(self.top_widget)
        self.top_grid_layout = Qt.QGridLayout()
        self.top_layout.addLayout(self.top_grid_layout)

        self.settings = Qt.QSettings("GNU Radio", "top_block")
        self.restoreGeometry(self.settings.value("geometry").toByteArray())

        ##################################################
        # Variables
        ##################################################
        self.tx_gain = tx_gain = 0
        self.rx_gain = rx_gain = 20
        self.num_taps = num_taps = [1] * 25
        self.freq = freq = 910e6
        self.decim = decim = 5
        self.dac_rate = dac_rate = 1e6
        self.ampl = ampl = 0.1
        self.adc_rate = adc_rate = 2e6

        ##################################################
        # Blocks
        ##################################################
        self.rfid_tag_decoder_0 = rfid.tag_decoder(int(adc_rate/decim))
        self.rfid_reader_0 = rfid.reader(int(adc_rate/decim), int(dac_rate))
        self.rfid_gate_0 = rfid.gate(int(adc_rate / decim))
        self.fir_filter_xxx_0 = filter.fir_filter_ccc(decim, (num_taps))
        self.fir_filter_xxx_0.declare_sample_delay(0)
        self.blocks_throttle_0 = blocks.throttle(gr.sizeof_gr_complex*1, 3 * int(adc_rate/decim),True)
        self.blocks_multiply_const_xx_0 = blocks.multiply_const_ff(ampl)
        self.blocks_float_to_complex_0 = blocks.float_to_complex(1)
        self.blocks_file_source_0 = blocks.file_source(gr.sizeof_gr_complex*1, '/home/andreas/PartIIB/Repos/rfid/Gen2-UHF-RFID-Reader/gr-rfid/misc/data/file_source_test', False)
        self.blocks_file_sink_2 = blocks.file_sink(gr.sizeof_gr_complex*1, '/home/andreas/PartIIB/Repos/rfid/Gen2-UHF-RFID-Reader/gr-rfid/misc/data/decoder', False)
        self.blocks_file_sink_2.set_unbuffered(False)
        self.blocks_file_sink_1 = blocks.file_sink(gr.sizeof_gr_complex*1, '/home/andreas/PartIIB/Repos/rfid/Gen2-UHF-RFID-Reader/gr-rfid/misc/data/matched_filter', False)
        self.blocks_file_sink_1.set_unbuffered(False)
        self.blocks_file_sink_0 = blocks.file_sink(gr.sizeof_gr_complex*1, '/home/andreas/PartIIB/Repos/rfid/Gen2-UHF-RFID-Reader/gr-rfid/misc/data/file_sink', False)
        self.blocks_file_sink_0.set_unbuffered(False)

        ##################################################
        # Connections
        ##################################################
        self.connect((self.blocks_file_source_0, 0), (self.fir_filter_xxx_0, 0))
        self.connect((self.blocks_float_to_complex_0, 0), (self.blocks_file_sink_0, 0))
        self.connect((self.blocks_multiply_const_xx_0, 0), (self.blocks_float_to_complex_0, 0))
        self.connect((self.blocks_throttle_0, 0), (self.rfid_gate_0, 0))
        self.connect((self.fir_filter_xxx_0, 0), (self.blocks_file_sink_1, 0))
        self.connect((self.fir_filter_xxx_0, 0), (self.blocks_throttle_0, 0))
        self.connect((self.rfid_gate_0, 0), (self.rfid_tag_decoder_0, 0))
        self.connect((self.rfid_reader_0, 0), (self.blocks_multiply_const_xx_0, 0))
        self.connect((self.rfid_tag_decoder_0, 1), (self.blocks_file_sink_2, 0))
        self.connect((self.rfid_tag_decoder_0, 0), (self.rfid_reader_0, 0))

    def closeEvent(self, event):
        self.settings = Qt.QSettings("GNU Radio", "top_block")
        self.settings.setValue("geometry", self.saveGeometry())
        event.accept()

    def get_tx_gain(self):
        return self.tx_gain

    def set_tx_gain(self, tx_gain):
        self.tx_gain = tx_gain

    def get_rx_gain(self):
        return self.rx_gain

    def set_rx_gain(self, rx_gain):
        self.rx_gain = rx_gain

    def get_num_taps(self):
        return self.num_taps

    def set_num_taps(self, num_taps):
        self.num_taps = num_taps
        self.fir_filter_xxx_0.set_taps((self.num_taps))

    def get_freq(self):
        return self.freq

    def set_freq(self, freq):
        self.freq = freq

    def get_decim(self):
        return self.decim

    def set_decim(self, decim):
        self.decim = decim
        self.blocks_throttle_0.set_sample_rate(3 * int(self.adc_rate/self.decim))

    def get_dac_rate(self):
        return self.dac_rate

    def set_dac_rate(self, dac_rate):
        self.dac_rate = dac_rate

    def get_ampl(self):
        return self.ampl

    def set_ampl(self, ampl):
        self.ampl = ampl
        self.blocks_multiply_const_xx_0.set_k(self.ampl)

    def get_adc_rate(self):
        return self.adc_rate

    def set_adc_rate(self, adc_rate):
        self.adc_rate = adc_rate
        self.blocks_throttle_0.set_sample_rate(3 * int(self.adc_rate/self.decim))


def main(top_block_cls=top_block, options=None):

    from distutils.version import StrictVersion
    if StrictVersion(Qt.qVersion()) >= StrictVersion("4.5.0"):
        style = gr.prefs().get_string('qtgui', 'style', 'raster')
        Qt.QApplication.setGraphicsSystem(style)
    qapp = Qt.QApplication(sys.argv)

    tb = top_block_cls()
    tb.start()
    tb.show()

    def quitting():
        tb.stop()
        tb.wait()
    qapp.connect(qapp, Qt.SIGNAL("aboutToQuit()"), quitting)
    qapp.exec_()


if __name__ == '__main__':
    main()
