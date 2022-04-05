### Steps :
1. Encoders with same architecture for every type of event (+ decoders). In the instantiation, only the input size varies.
2. Basic clustering types (static e.g. t-SNE): 
    * directly on input
    * on output of encoders
    * on computed history
    * with/without the decoders
    * depending on the losses used during training (e.g. with/without specific clustering loss)
    * ...

