from translate import Translate

trans = Translate()
encoder_input = trans.convert2ind('Lâche')
encoder_output, (hidden_state, cell_state) = trans.encoder(encoder_input)

decoder_input = trans.convert2ind('L')
attn_decoder = trans.attn_decoder
output = attn_decoder(decoder_input, hidden_state, encoder_output)
