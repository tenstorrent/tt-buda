import torch


class ViLtEmbeddingWrapper(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.vilt_model = model

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        pixel_values=None,
        pixel_mask=None,
        inputs_embeds=None,
        image_embeds=None,
        image_token_type_idx=None,
    ):

        embeddings, masks = self.vilt_model.vilt.embeddings(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            pixel_values=pixel_values,
            pixel_mask=pixel_mask,
            inputs_embeds=inputs_embeds,
            image_embeds=image_embeds,
            image_token_type_idx=image_token_type_idx,
        )
        return embeddings, masks


class ViltModelWrapper(torch.nn.Module):
    def __init__(self, model, task=None, text_seq_len=None):
        super().__init__()
        self.vilt_model = model
        self.task = task
        self.text_seq_len = text_seq_len

    def forward(self, embedding_output, attention_mask, head_mask=None):

        head_mask = self.vilt_model.vilt.get_head_mask(head_mask, self.vilt_model.vilt.config.num_hidden_layers)

        extended_attention_mask = attention_mask[:, None, None, :]
        extended_attention_mask = (1.0 - extended_attention_mask) * torch.finfo(torch.float32).min

        encoder_outputs = self.vilt_model.vilt.encoder(
            embedding_output,
            attention_mask=extended_attention_mask,
            head_mask=head_mask,
            return_dict=False,
        )

        sequence_output = encoder_outputs[0]

        sequence_output = self.vilt_model.vilt.layernorm(sequence_output)
        pooled_output = (
            self.vilt_model.vilt.pooler(sequence_output) if self.vilt_model.vilt.pooler is not None else None
        )

        viltmodel_output = (sequence_output, pooled_output) + encoder_outputs[1:]

        sequence_output, pooled_output = viltmodel_output[:2]

        if self.task == "maskedlm":

            if self.text_seq_len is None:
                raise ValueError("You cannot must provide text sequence length")

            text_features, _ = (sequence_output[:, : self.text_seq_len], sequence_output[:, self.text_seq_len :])

            mlm_logits = self.vilt_model.mlm_score(text_features)

            viltmodel_output = (mlm_logits,) + viltmodel_output[2:]

        if self.task == "qa":

            logits = self.vilt_model.classifier(pooled_output)

            viltmodel_output = (logits,) + viltmodel_output[2:]

        return viltmodel_output
