package v1alpha1

import (
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/runtime"
)

// PrewarmRequestSpec describes the desired prewarming state.
type PrewarmRequestSpec struct {
	Tool            string      `json:"tool"`
	Service         string      `json:"service"`
	Probability     float64     `json:"probability"`
	Threshold       float64     `json:"threshold"`
	TTLSeconds      int64       `json:"ttlSeconds"`
	DesiredReplicas int32       `json:"desiredReplicas"`
	RequestedAt     metav1.Time `json:"requestedAt"`
	EpisodeID       string      `json:"episodeId,omitempty"`
	WorkflowID      string      `json:"workflowId,omitempty"`
}

// PrewarmRequestStatus describes the observed prewarming state.
type PrewarmRequestStatus struct {
	Phase          string       `json:"phase,omitempty"`
	ReadyAt        *metav1.Time `json:"readyAt,omitempty"`
	ExpiresAt      *metav1.Time `json:"expiresAt,omitempty"`
	LastActivityAt *metav1.Time `json:"lastActivityAt,omitempty"`
	Message        string       `json:"message,omitempty"`
}

// PrewarmRequest represents a request to prewarm a service.
type PrewarmRequest struct {
	metav1.TypeMeta   `json:",inline"`
	metav1.ObjectMeta `json:"metadata,omitempty"`
	Spec              PrewarmRequestSpec   `json:"spec"`
	Status            PrewarmRequestStatus `json:"status,omitempty"`
}

// PrewarmRequestList contains a list of PrewarmRequest objects.
type PrewarmRequestList struct {
	metav1.TypeMeta `json:",inline"`
	metav1.ListMeta `json:"metadata,omitempty"`
	Items           []PrewarmRequest `json:"items"`
}

func init() {
	SchemeBuilder.Register(&PrewarmRequest{}, &PrewarmRequestList{})
}

// DeepCopyObject returns a deep copy of the PrewarmRequest.
func (in *PrewarmRequest) DeepCopyObject() runtime.Object {
	if in == nil {
		return nil
	}
	out := new(PrewarmRequest)
	in.DeepCopyInto(out)
	return out
}

// DeepCopyInto copies the receiver into out.
func (in *PrewarmRequest) DeepCopyInto(out *PrewarmRequest) {
	*out = *in
	out.ObjectMeta = *in.ObjectMeta.DeepCopy()
	out.Spec = in.Spec
	out.Status = in.Status

	if in.Status.ReadyAt != nil {
		out.Status.ReadyAt = in.Status.ReadyAt.DeepCopy()
	}
	if in.Status.ExpiresAt != nil {
		out.Status.ExpiresAt = in.Status.ExpiresAt.DeepCopy()
	}
	if in.Status.LastActivityAt != nil {
		out.Status.LastActivityAt = in.Status.LastActivityAt.DeepCopy()
	}
}

// DeepCopyObject returns a deep copy of the PrewarmRequestList.
func (in *PrewarmRequestList) DeepCopyObject() runtime.Object {
	if in == nil {
		return nil
	}
	out := new(PrewarmRequestList)
	in.DeepCopyInto(out)
	return out
}

// DeepCopyInto copies the receiver into out.
func (in *PrewarmRequestList) DeepCopyInto(out *PrewarmRequestList) {
	*out = *in
	out.ListMeta = *in.ListMeta.DeepCopy()

	if in.Items != nil {
		out.Items = make([]PrewarmRequest, len(in.Items))
		for i := range in.Items {
			in.Items[i].DeepCopyInto(&out.Items[i])
		}
	}
}
