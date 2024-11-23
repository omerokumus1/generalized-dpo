pip install -r https://raw.githubusercontent.com/rasbt/LLMs-from-scratch/main/requirements.txt


-> If you perform operations inside a torch.no_grad() block, 
    the computation graph will not be created, even if requires_grad=True
    
-> If you use the .detach() method on a tensor, it will create a new tensor 
    that shares the same data but is detached from the computation graph


# TODO
    +> Try another dummy loss function
        RuntimeError: one of the variables needed for gradient computation has been modified by an inplace operation: 
        [torch.cuda.FloatTensor [2, 203, 2048]], which is output 0 of MulBackward0, is at version 1; expected 
        version 0 instead. Hint: the backtrace further above shows the operation that failed to compute its gradient. 
        The variable in question was changed in there or anywhere later. Good luck!
        
        ****XXX Asıl sorun 2 kere forward pass olması. 
            - tek forward pass yap, output'tan chosen we rejected'ları çekmeye çalış

        - Aşağıda # satırlarını dahil edince hata veriyor.
            chosen_logits = model(batch["chosen"]).logits
            rejected_logits = model(batch["rejecteds"][0][:2]).logits
            
            #chosen_logits_ref = reference_model(batch["chosen"]).logits
            #rejected_logits_ref = reference_model(batch["rejecteds"][0][:2]).logits

    -> rejecteds için nan değer geliyor. Bu da sıkıntı yaratıyor. NaN değeri handle et.

    -> BatchNorm and Dropout layers

    -> can my DataLoader affect the computation graph?
    -> Move over computation graph
    
    -> load_llm datatype = None olunca hata veriyor. Bununla ilgili bir şeyler yap
    -> handle nan and inf values in the data

# ProcessedBatch Types
    -> prompt: List[Tensor]
    ->

# Notes
    ->X Asıl sorun loss ile optimizer arasındaki bağlantının olmaması
        - https://stackoverflow.com/questions/53975717/pytorch-connection-between-loss-backward-and-optimizer-step

    -> get_max_of_rejected_logprobs çıktısı yanlış. yeni bir tensör oluşturmadan bu işi halletmen lazım
        **direkt rejected_log_probas_list döndürmeyi dene
        **eski unsloth ile devam et
        çıktılar:
            rejected_log_probas_list: [tensor([-0.1599, -0.2587, -0.5238], device='cuda:0', grad_fn=<DivBackward0>), tensor([-0.1936, -0.2239, -0.1717], device='cuda:0', grad_fn=<DivBackward0>)]
            rejected_log_probas_list[0].shape: torch.Size([3])
            rejected_log_probas_list[1].shape: torch.Size([3])
            rejected_log_probas_list[0]: tensor(-0.1599, device='cuda:0', grad_fn=<MaxBackward1>)
            rejected_log_probas_list[1]: tensor(-0.1717, device='cuda:0', grad_fn=<MaxBackward1>)
            rejected_log_probas_list[0].requires_grad: True
            rejected_log_probas_list[1].requires_grad: True
            result: tensor([-0.1599, -0.1717], device='cuda:0')
            result.shape: torch.Size([2])
            result.requires_grad: False

            **> yeni tensör oluşturduğun için requires_grad False oldu ve grad_fn'yi kaybettin.
                rejected_log_probas_list[0] ve rejected_log_probas_list[1] tensörlerini birleştirip yeni bir tensör oluşturmadan döndürmeyi dene

        chosen için çıktılar:
        result: tensor([-2.6703, -2.6807], device='cuda:0', grad_fn=<DivBackward0>)
        result.shape: torch.Size([2])
        result.requires_grad: True

    -> All above is bullshit
    -> use eq function of ProcessedBatch for each function's head and tail to spot in which function the in-place operation occurs


    -> ChatGPT son 2 cevaba bak
    -> forward pass'te sorun olabilir. Onu kontrol et
    -> ChatGPT'ye torch ile dpo yazdır, onu manipüle etmeyi dene

    -> train_model_dpo_simple içinden compute_dpo_loss_batch fonk tamamen kaldırdım ama yine de hata devam ediyor. Demek ki
        compute_dpo_loss_batch fonksiyonu ile ilgili bir sorun yok.
    -> forward pass ile ya da optimizer ile ilgili bir sorun olabilir

    -> Sorun yüksek ihtimalle forward pass'te gerçekleşiyor. Data loading ve preprocessing adımlarını
    kontrol etmek lazım

    -> Batch tensor'leri requires_grad=False. Bu yüzden loss hesaplanırken grad_fn oluşmuyor ve computation graph'e dahil olmuyor

# DEBUG
    ipdb> losses.grad_fn
    <NegBackward0 object at 0x79ac34be8c70>
    ipdb> logits.grad_fn
    <SubBackward0 object at 0x79ac34bebee0>
    ipdb> model_logratios.grad_fn
    <SubBackward0 object at 0x79ac34be9540>
    ipdb> policy_chosen_logprobs.grad_fn
    <DivBackward0 object at 0x79ac34beb430>
    ipdb> policy_rejected_logprobs.grad_fn
    <DivBackward0 object at 0x79ac34beaa70>
    ipdb> reference_chosen_logprobs.grad_fn
    <DivBackward0 object at 0x79ac34bea7a0>
    ipdb> reference_rejected_logprobs.grad_fn
    <DivBackward0 object at 0x79ac34bebee0>
    ipdb> losses
    tensor([0.6914, 0.6914], device='cuda:0', dtype=torch.bfloat16,
           grad_fn=<NegBackward0>)
    ipdb> logits
    tensor([0., 0.], device='cuda:0', dtype=torch.bfloat16, grad_fn=<SubBackward0>)
    
    *** ipdb> (beta*logits).grad_fn
    *** <MulBackward0 object at 0x79ac2dd30a90>
    
    ipdb> logsigmoid(beta * logits).grad_fn
    *** NameError: name 'logsigmoid' is not defined
    ipdb> torch.nn.functional.F.logsigmoid(beta * logits).grad_fn
    *** AttributeError: module 'torch.nn.functional' has no attribute 'F'
    ipdb> torch.nn.functional.logsigmoid(beta * logits).grad_fn
    <LogSigmoidBackward0 object at 0x79ac2dd30b20>
    ipdb> policy_chosen_logprobs
    tensor([-2.2344, -2.7344], device='cuda:0', dtype=torch.bfloat16,
           grad_fn=<DivBackward0>)
    ipdb> policy_rejected_logprobs
    tensor([-0.2432, -0.3086], device='cuda:0', dtype=torch.bfloat16,
           grad_fn=<DivBackward0>)
    ipdb> reference_chosen_logprobs
    tensor([-2.2344, -2.7344], device='cuda:0', dtype=torch.bfloat16,
           grad_fn=<DivBackward0>)
    ipdb> reference_rejected_logprobs
    tensor([-0.2432, -0.3086], device='cuda:0', dtype=torch.bfloat16,
           grad_fn=<DivBackward0>)