# coding=utf-8
# copied from bart

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.model_utils import BaseModel
from transformers.generation_utils import top_k_top_p_filtering
from transformers.models.blenderbot_small import (BlenderbotSmallConfig, BlenderbotSmallForConditionalGeneration,)
from transformers.modeling_outputs import (BaseModelOutput, Seq2SeqModelOutput, Seq2SeqLMOutput,)
# from .PARAMS import SAMPLE, TEMPERATURE
from torch.nn.utils.rnn import pad_sequence
# from encoder import UtteranceEncoder
# from transrgat import TRANSRGAT
from transgraph import TRANSGRAPH
# from decoder import UtteranceDecoder

# class GatedMultimodalLayer(nn.Module):
#     """ Gated Multimodal Layer based on 'Gated multimodal networks, Arevalo1 et al.' (https://arxiv.org/abs/1702.01992) """
#     def __init__(self, size_in1, size_in2, size_out):
#         super(GatedMultimodalLayer, self).__init__()
#         self.size_in1, self.size_in2, self.size_out = size_in1, size_in2, size_out

#         self.hidden1 = nn.Linear(size_in1, size_out, bias=False)
#         self.hidden2 = nn.Linear(size_in2, size_out, bias=False)
#         self.hidden_sigmoid = nn.Linear(size_out * 2, 1, bias=False)

#         # Activation functions
#         self.tanh_f = nn.Tanh()
#         self.sigmoid_f = nn.Sigmoid()

#     def forward(self, x1, x2):
#         h1 = self.tanh_f(self.hidden1(x1))
#         h2 = self.tanh_f(self.hidden2(x2))
#         x = torch.cat((h1, h2), dim=1)
#         z = self.sigmoid_f(self.hidden_sigmoid(x))

#         return z.view(z.size()[0], 1) * h1 + (1-z).view(z.size()[0], 1) * h2
    

class Model(BaseModel, BlenderbotSmallForConditionalGeneration):
    def __init__(self, config: BlenderbotSmallConfig, 
                 in_channels=512,out_channels=100, num_relations=4, heads=4, num_layers=2):
        super().__init__(config)

        # self.transrgat = TRANSRGAT(in_channels, out_channels, num_relations, heads, num_layers)
        self.transgraph = TRANSGRAPH(in_channels, out_channels, num_relations, heads, num_layers)
        self.att1 = nn.MultiheadAttention(embed_dim=in_channels, num_heads=2,batch_first=True)
        self.att2 = nn.MultiheadAttention(embed_dim=in_channels, num_heads=2,batch_first=True)

        self.mlp_stra = nn.Linear(in_channels,8)
        self.mlp_emo = nn.Linear(in_channels,7)

        self.strat_embedding = nn.Embedding(8, 512)
        # self.gate_unit = GatedMultimodalLayer(512, 512, 512)
        # self.LayerNorm = nn.LayerNorm(normalized_shape = 512)

        self.mlp_sit = nn.Linear(768,in_channels)
        self.att_fusion = nn.Linear(512*3,512)
        self.sit_fusion = nn.Linear(in_channels * 2, in_channels)


    def forward(self, input_ids, attention_mask, cls_indices, edge_index,edge_type,edge_repre,\
                decoder_input_ids,decoder_attention_mask,labels,strat_id,stra_id_his,emotion_id_his,  \
                problems, situations, encoder_outputs=None, past_key_values=None, validation=False, **kwargs):
        if self.training or validation:
            enc_emb = self.model.encoder(input_ids, attention_mask)['last_hidden_state']
            graph_input = []
            for i,cls in enumerate(cls_indices):
                for j in cls:
                    graph_input.append(enc_emb[i,j,:])
            graph_input = torch.stack(graph_input,dim=0)
            # graph_output= self.transrgat(x=graph_input, edge_index=edge_index, edge_type=edge_type,edge_repre=edge_repre)
            graph_output= self.transgraph(x=graph_input, edge_index=edge_index, edge_type=edge_type,edge_repre=edge_repre)

            graph_stra = self.mlp_stra(graph_output)
            stra_id_his = torch.tensor(sum(stra_id_his,[])).cuda()
            loss_stra = F.cross_entropy(graph_stra, stra_id_his, reduction='mean')

            graph_emo = self.mlp_emo(graph_output)
            emotion_id_his = torch.tensor(sum(emotion_id_his,[])).cuda()
            loss_emo = F.cross_entropy(graph_emo, emotion_id_his, reduction='mean')


            graph_out = []
            graph_last_strat = []
            start = 0
            end = 0
            for cls in cls_indices:
                end = end + len(cls)
                graph_out.append(graph_output[start:end,:])
                graph_last_strat.append(graph_stra[end-1,:])
                start = end

            graph_att = pad_sequence([torch.tensor([1.] * g.size(0), dtype=torch.float) for g in graph_out],batch_first=True, padding_value=0.).cuda()
            graph_out = pad_sequence(graph_out, batch_first=True, padding_value=0)

            graph_last_strat = torch.stack(graph_last_strat)
              
            equal_stra = (torch.argmax(graph_last_strat,dim=1) == torch.tensor(strat_id).cuda()).sum().item()

            mixed_stra = torch.matmul(F.softmax(graph_last_strat,dim=1), self.strat_embedding.weight).unsqueeze(1)

            
            attn_output1, _ = self.att1(query=enc_emb,key=graph_out,value=graph_out,key_padding_mask=graph_att)

            attn_output2, _ = self.att2(query=enc_emb,key=mixed_stra,value=mixed_stra)

            # fused_attn = []
            # for i in range(enc_emb.shape[1]):
            #     fused_attn.append(self.gate_unit(attn_output1[:, i, :], attn_output2[:, i, :])) 
            
            # attn_output = torch.stack(fused_attn,dim=1) + enc_emb
            # attn_output = self.LayerNorm(attn_output)
            g = nn.Sigmoid()(self.att_fusion(torch.cat([attn_output1, attn_output2,enc_emb], dim=-1)))
            attn_output = g * attn_output1 + (1-g) * attn_output2 + enc_emb
            # attn_output = attn_output1 + attn_output2 + enc_emb
            
            assert self.toker is not None
            decoder_input_ids = decoder_input_ids.view(-1, decoder_input_ids.size()[-1])
            inputs_embeds = self.model.get_decoder().embed_tokens(decoder_input_ids) * self.model.get_decoder().embed_scale
            if situations is not None:
                gate = nn.Sigmoid()(self.sit_fusion(torch.cat([inputs_embeds, self.mlp_sit(situations).unsqueeze(1).expand_as(inputs_embeds)], dim=-1)))
                inputs_embeds = gate * inputs_embeds + (1-gate) * self.mlp_sit(situations).unsqueeze(1).expand_as(inputs_embeds)
            
            outputs = self.model.decoder(
                inputs_embeds = inputs_embeds,
                attention_mask = decoder_attention_mask,
                encoder_hidden_states = attn_output,
                encoder_attention_mask = attention_mask,
                use_cache = True,
                return_dict = True,
            )
            lm_logits = self.lm_head(outputs[0]) + self.final_logits_bias
        
            masked_lm_loss = None
            if labels is not None:
                loss = F.cross_entropy(lm_logits.view(-1, lm_logits.size(-1)), labels.view(-1), reduction='none')
                loss = loss.view(labels.size(0), labels.size(1))
                label_size = torch.sum(labels.ne(-100), dim=1).type_as(loss)
                masked_lm_loss = torch.sum(loss) / torch.sum(label_size)
                ppl_value = torch.exp(torch.mean(torch.sum(loss, dim=1).float() / label_size.float()))


            if self.training and validation == False: # training
                res = {'all': masked_lm_loss + loss_stra + loss_emo, 'ppl': ppl_value, }
                return res
            elif self.training == False and validation == True: # validation
                return loss, label_size, equal_stra, len(strat_id)
        else:   # infer
            assert decoder_input_ids.size(1) == 1
            assert labels == None
            assert self.toker is not None

            attn_output = encoder_outputs.last_hidden_state
            
            decoder_input_ids = decoder_input_ids.view(-1, decoder_input_ids.size()[-1])
            inputs_embeds = self.model.get_decoder().embed_tokens(decoder_input_ids) * self.model.get_decoder().embed_scale
            if situations is not None:
                gate = nn.Sigmoid()(self.sit_fusion(torch.cat([inputs_embeds, self.mlp_sit(situations).unsqueeze(1).expand_as(inputs_embeds)], dim=-1)))
                inputs_embeds = gate * inputs_embeds + (1-gate) * self.mlp_sit(situations).unsqueeze(1).expand_as(inputs_embeds)
            
            outputs = self.model.decoder(
                inputs_embeds = inputs_embeds,
                attention_mask = decoder_attention_mask,
                encoder_hidden_states = attn_output,
                encoder_attention_mask = attention_mask,
                past_key_values=past_key_values,
                use_cache = True,
                return_dict = True,
            )
            lm_logits = self.lm_head(outputs[0]) + self.final_logits_bias

            return Seq2SeqLMOutput(
                # loss=None,
                logits=lm_logits,
                past_key_values=outputs.past_key_values,
                # decoder_hidden_states=outputs.decoder_hidden_states,
                # decoder_attentions=outputs.decoder_attentions,
                # cross_attentions=outputs.cross_attentions,
                # encoder_last_hidden_state=outputs.encoder_last_hidden_state,
                # encoder_hidden_states=outputs.encoder_hidden_states,
                # encoder_attentions=outputs.encoder_attentions,
            )
        
    def prepare_inputs_for_generation(
        self,
        decoder_input_ids,
        past=None,
        attention_mask=None,
        head_mask=None,
        decoder_head_mask=None,
        cross_attn_head_mask=None,
        use_cache=None,
        encoder_outputs=None,
        **kwargs
    ):
        # cut decoder_input_ids if past is used
        if past is not None:
            decoder_input_ids = decoder_input_ids[:, -1:]

        return {
            "input_ids": None,  # encoder_outputs is defined. input_ids not needed
            "encoder_outputs": encoder_outputs,
            "past_key_values": past,
            "decoder_input_ids": decoder_input_ids,
            "attention_mask": attention_mask,
            "head_mask": head_mask,
            "decoder_head_mask": decoder_head_mask,
            "cross_attn_head_mask": cross_attn_head_mask,
            "use_cache": use_cache,  # change this to avoid caching (presumably for debugging)
            **kwargs
        }


    @torch.no_grad()
    def generate(self, input_ids, attention_mask, decoder_input_ids, **kwargs):
        cls_indices = kwargs['cls_indices']
        edge_index = kwargs['edge_index']
        edge_type = kwargs['edge_type']
        edge_repre = kwargs['edge_repre']
        
        encoder_outputs = self.model.encoder(input_ids, attention_mask)
        enc_emb = encoder_outputs['last_hidden_state']
        graph_input = []
        for i,cls in enumerate(cls_indices):
            for j in cls:
                graph_input.append(enc_emb[i,j,:])
        graph_input = torch.stack(graph_input,dim=0)
        # graph_output= self.transrgat(x=graph_input, edge_index=edge_index, edge_type=edge_type,edge_repre=edge_repre) 
        graph_output= self.transgraph(x=graph_input, edge_index=edge_index, edge_type=edge_type,edge_repre=edge_repre)

        graph_stra = self.mlp_stra(graph_output)
        
        graph_out = []
        graph_last_strat = []
        start = 0
        end = 0
        for cls in cls_indices:
            end = end + len(cls)
            graph_out.append(graph_output[start:end,:])
            graph_last_strat.append(graph_stra[end-1,:])
            start = end

        graph_att = pad_sequence([torch.tensor([1.] * g.size(0), dtype=torch.float) for g in graph_out],batch_first=True, padding_value=0.).cuda()
        graph_out = pad_sequence(graph_out, batch_first=True, padding_value=0)
        
        graph_last_strat = torch.stack(graph_last_strat)
        mixed_stra = torch.matmul(F.softmax(graph_last_strat,dim=1), self.strat_embedding.weight).unsqueeze(1)
       
        attn_output1, _ = self.att1(query=enc_emb,key=graph_out,value=graph_out,key_padding_mask=graph_att)
        attn_output2, _ = self.att2(query=enc_emb,key=mixed_stra,value=mixed_stra)

        g = nn.Sigmoid()(self.att_fusion(torch.cat([attn_output1, attn_output2,enc_emb], dim=-1)))
        attn_output = g * attn_output1 + (1-g) * attn_output2 + enc_emb       
        # attn_output = 0.1 * attn_output1 + 0.1 * attn_output2 + enc_emb


        encoder_outputs.last_hidden_state = attn_output
        
        
        generations = super().generate(
            attention_mask=attention_mask,
            encoder_outputs=encoder_outputs,
            decoder_input_ids=decoder_input_ids,
            **kwargs
        )
        return generations[:, 1:]
        

        
        
       
        
    
   